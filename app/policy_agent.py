from pathlib import Path
from dotenv import load_dotenv
import os
import re
import math
import pdfplumber
import numpy as np
from openai import OpenAI
from typing import List


class PolicyAgent:
    print("Loaded policy_agent from:", __file__)
    def __init__(self, pdf_paths: List[str], embedding_model: str = "text-embedding-3-small"):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment (.env).")
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

        # Storage
        self.chunks: List[str] = []
        self.embeddings = None  # numpy array of shape (n_chunks, dim)

        # Extract text and tables into chunks
        self._extract_and_chunk(pdf_paths)

        if len(self.chunks) == 0:
            raise RuntimeError("No chunks extracted from PDFs. Check files and extraction logic.")

        # Generate embeddings (batched)
        self._embed_chunks()
        print(f"Processed {len(self.chunks)} chunks.")

    # -------------------------
    # Extraction + chunking
    # -------------------------
    def _extract_and_chunk(self, pdf_paths: List[str]) -> None:
        """
        Extract text + tables from each PDF page using pdfplumber,
        split page text into paragraphs, and add table rows as table-text chunks.
        """
        chunks: List[str] = []

        for path in pdf_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                print(f"Warning: PDF path not found: {path}")
                continue

            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    # 1) Page text -> split into paragraphs by blank line
                    text = page.extract_text()
                    if text:
                        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
                        for p in paragraphs:
                            # Filter out tiny noise paragraphs
                            if len(p) >= 120:
                                chunks.append(p)

                    # 2) Extract tables (if any) and convert to readable pipe-separated rows
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            # table is a list of rows (lists)
                            row_texts = []
                            for row in table:
                                # replace None with empty string, strip whitespace
                                cells = [("" if cell is None else str(cell).strip()) for cell in row]
                                # join with pipe to keep column boundaries
                                row_text = " | ".join(cells)
                                row_texts.append(row_text)
                            table_text = "\n".join(row_texts)
                            # Keep only meaningful tables
                            if len(table_text) >= 60:
                                chunks.append("TABLE:\n" + table_text)

        # final filter and store
        # remove duplicates while preserving order
        seen = set()
        filtered = []
        for c in chunks:
            if c not in seen:
                filtered.append(c)
                seen.add(c)
        self.chunks = filtered

    # -------------------------
    # Embedding generation (batched)
    # -------------------------
    def _embed_chunks(self, batch_size: int = 10) -> None:
        """
        Generate embeddings for self.chunks and store them as a numpy array.
        Batches reduce per-request overhead.
        """
        all_embeddings = []
        n = len(self.chunks)
        for i in range(0, n, batch_size):
            batch = self.chunks[i : i + batch_size]
            # OpenAI embeddings endpoint accepts a list of inputs
            resp = self.client.embeddings.create(model=self.embedding_model, input=batch)
            # resp.data is a list aligned with batch
            for item in resp.data:
                all_embeddings.append(item.embedding)

        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        # normalize for cosine similarity convenience
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # avoid zero division
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms
    
    def rewrite_query(self, query):
     response = self.client.responses.create(
        model="gpt-4o-mini",
        input=f"""
        Rewrite the following question to maximize retrieval from an insurance policy document.
        Keep it concise.

         Question:
         {query} """)
     return response.output_text.strip()


    def retrieve_relevant_chunks(self, query, top_k=10):
      resp = self.client.embeddings.create(
        model=self.embedding_model,
        input=query
        )

      query_embedding = np.array(resp.data[0].embedding)
      query_embedding = query_embedding / np.linalg.norm(query_embedding)

      similarities = self.embeddings.dot(query_embedding)

      top_indices = similarities.argsort()[-top_k:][::-1]

      results = []
      for idx in top_indices:
        results.append({
            "chunk": self.chunks[idx],
            "score": float(similarities[idx])
        })
      return results



    # # -------------------------
    # # Retrieval (embedding cosine similarity)
    # # -------------------------
    # def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> List[str]:
    #     """
    #     Return top_k most similar chunks to the query using cosine similarity.
    #     """
    #     # embed query
    #     resp = self.client.embeddings.create(model=self.embedding_model, input=query)
    #     q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    #     q_norm = np.linalg.norm(q_emb)
    #     if q_norm == 0:
    #         q_norm = 1.0
    #     q_emb = q_emb / q_norm

    #     # compute cosine similarities
    #     sims = self.embeddings.dot(q_emb)  # shape (n_chunks,)
    #     # get top indices
    #     top_k = min(top_k, len(self.chunks))
    #     top_indices = np.argsort(sims)[-top_k:][::-1]
    #     top_chunks = [self.chunks[i] for i in top_indices]
    #     top_scores = [float(sims[i]) for i in top_indices]

    #     # return chunks (optionally could return scores)
    #     return top_chunks

    # -------------------------
    # Reranking with LLM (robust parsing)
    # -------------------------
    def rerank_chunks(self, query: str, candidate_chunks: List[str], max_tokens: int = 64) -> List[str]:
        """
        Use an LLM call to score / rank candidate chunks.
        Returns an ordered list (best first). This implementation asks the model to return
        a JSON array of indices in ranked order for robustness.
        """
        if not candidate_chunks:
            return []

        # Build a compact prompt that asks for JSON array of indices
        prompt = "You are a retrieval assistant. Given the question and a list of candidate text chunks,"
        prompt += " return a JSON array of indices (0-based) ordering the chunks from most to least likely to contain the answer.\n\n"
        prompt += f"Question:\n{query}\n\nChunks:\n"
        for i, c in enumerate(candidate_chunks):
            short = c if len(c) < 500 else c[:500] + " ...[truncated]..."
            prompt += f"\n[{i}] {short}\n"

        prompt += (
            "\nRespond ONLY with a JSON array like: [0,2,1]\n"
            "Where the first index is the chunk most likely to contain the answer.\n"
            "If none likely, you may return an empty array [].\n"
        )

        # call LLM for reranking (ensure min output tokens >= 16)
        resp = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=max(32, max_tokens),
        )

        text = resp.output_text.strip()

        # extract first JSON array from the response robustly
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if not m:
            # fallback: return original order
            return candidate_chunks

        arr_text = m.group()
        try:
            # safe parse of integers
            indices = [int(re.sub(r"[^\d]", "", s)) for s in re.findall(r"\d+", arr_text)]
        except Exception:
            return candidate_chunks

        # clamp indices and build ranked chunks
        ranked = []
        for idx in indices:
            if 0 <= idx < len(candidate_chunks):
                ranked.append(candidate_chunks[idx])

        # append any remaining (not included) in original order
        for i, c in enumerate(candidate_chunks):
            if c not in ranked:
                ranked.append(c)

        return ranked

    # -------------------------
    # Answer query (retrieve + rerank + answer)
    # -------------------------
    def answer_query(self, prompt: str, retrieve_k: int = 12, use_top_n: int = 4) -> str:
        """
        Full pipeline:
        - retrieve retrieve_k candidates by embedding similarity
        - rerank candidates using LLM, take top `use_top_n`
        - send context + question to responses.create() and return output_text
        """
        # 1) retrieve candidates
        rewritten_query = self.rewrite_query(prompt)
        candidates = self.retrieve_relevant_chunks(rewritten_query,top_k=retrieve_k)
        #candidates = self.retrieve_relevant_chunks(prompt, top_k=retrieve_k)
        if not candidates:
            return "I don't know."

        # 2) rerank (returns list best-first)
        ranked = self.rerank_chunks(prompt, candidates)

        # 3) choose top N (safely)
        # top_n = ranked[:max(1, min(use_top_n, len(ranked)))]
        # context = "\n\n---\n\n".join(top_n)

        top_n = ranked[:max(1, min(use_top_n, len(ranked)))]

        # Extract text only if items are dicts
        if isinstance(top_n[0], dict):
           context = "\n\n---\n\n".join([item["chunk"] for item in top_n])
        else:
           context = "\n\n---\n\n".join(top_n)

        # Helpful debug output (can be removed)
        # print("\n--- Retrieved / Reranked Chunks (top) ---\n")
        # for i, c in enumerate(top_n):
        #     print(f"--- chunk #{i} (len={len(c)} chars) ---")
        #     print(c[:800])
        #     print("\n------------------\n")

        print("\n--- Retrieved / Reranked Chunks (top) ---\n")
        for i, c in enumerate(top_n):
            if isinstance(c, dict):
               chunk_text = c["chunk"]
            else:
               chunk_text = c

            print(f"--- chunk #{i} (len={len(chunk_text)} chars) ---")
            print(chunk_text[:800])
            print("\n------------------\n")

        # 4) Ask the model, instruct to answer ONLY from context or reply "I don't know"
        system = """
        You are a policy question-answering assistant.

        Answer ONLY using the provided CONTEXT.
        Do NOT use outside knowledge.
        Do NOT infer beyond what is written.
        If the answer is not explicitly present in the CONTEXT, respond with exactly:

        I don't know.
        """

        user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}\n\nProvide a concise answer."

        resp = self.client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_output_tokens=500,
        )

        answer = resp.output_text.strip()
        return answer

    def debug_answer_query(self, prompt, retrieve_k=12, use_top_n=4):

     rewritten_query = self.rewrite_query(prompt)

     candidates = self.retrieve_relevant_chunks(rewritten_query, retrieve_k)

     if not candidates:
        return {
            "answer": "I don't know.",
            "rewritten_query": rewritten_query,
            "chunks": []
        }

    # Extract text only for reranking
     chunk_texts = [c["chunk"] for c in candidates]

     ranked_texts = self.rerank_chunks(prompt, chunk_texts)

     ranked_results = []
     for text in ranked_texts[:use_top_n]:
        for c in candidates:
            if c["chunk"] == text:
                ranked_results.append(c)

     context = "\n\n---\n\n".join([r["chunk"] for r in ranked_results])

     resp = self.client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "Answer strictly using only the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"}
        ],
        max_output_tokens=300
    )

     return {
        "answer": resp.output_text.strip(),
        "rewritten_query": rewritten_query,
        "chunks": ranked_results
    }
    # -------------------------
    # Utility: optionally delete embeddings / free memory
    # -------------------------
    def clear_embeddings(self):
        self.embeddings = None

