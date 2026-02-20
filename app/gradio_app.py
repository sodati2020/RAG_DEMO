import gradio as gr
from app.policy_agent import PolicyAgent

agent = PolicyAgent([
    "data/kaiser_1.pdf",
    "data/Kaiser_2.pdf",
])

def respond(message, history):
    result = agent.debug_answer_query(message)

    debug_info = f"""
🧠 Rewritten Query:
{result['rewritten_query']}

📊 Retrieved Chunks with Similarity Scores:
----------------------------------------
"""

    for i, item in enumerate(result["chunks"]):
        debug_info += (
            f"\nChunk {i+1} | Score: {item['score']:.4f}\n"
            f"{item['chunk'][:500]}\n\n"
        )

    return result["answer"], debug_info


with gr.Blocks() as demo:
    gr.Markdown("# 📄 Kaiser Policy QA (RAG Demo)")
    gr.Markdown("Custom RAG with embeddings + retrieval + reranking.")

    chat = gr.ChatInterface(
        fn=respond,
        additional_outputs=[
            gr.Textbox(label="🔍 Retrieval Debug Info", lines=15)
        ]
    )

demo.launch()