from app.policy_agent import PolicyAgent


agent = PolicyAgent(
    [
        "data/kaiser_1.pdf",
        "data/Kaiser_2.pdf",
    ]
)


# ----------------------------
# Helper functions
# ----------------------------

def text_matches(answer, expected_terms):
    answer_lower = answer.lower()
    return all(term.lower() in answer_lower for term in expected_terms)


def text_matches_any(answer, expected_terms):
    answer_lower = answer.lower()
    return any(term.lower() in answer_lower for term in expected_terms)


# ----------------------------
# Test cases
# ----------------------------

TEST_CASES = [

    # ----------------------------
    # TABLE RETRIEVAL TESTS
    # ----------------------------
    {
        "name": "Out-of-pocket maximum",
        "question": "What is the Plan Out-of-Pocket Maximum?",
        "expected_contains": ["$3,000", "$6,000"]
    },
    {
        "name": "Annual maximum wording variation",
        "question": "What is the yearly out-of-pocket cap?",
        "expected_contains": ["$3,000", "$6,000"]
    },

    # ----------------------------
    # DEDUCTIBLE TESTS
    # ----------------------------
    {
        "name": "Plan deductible",
        "question": "Does this plan have a deductible?",
        "expected_contains_any": ["None", "No"]
    },
    {
        "name": "Drug deductible",
        "question": "Is there a Drug Deductible?",
        "expected_contains_any": ["None", "No"]
    },

    # ----------------------------
    # DEFINITION TESTS
    # ----------------------------
    {
        "name": "Accumulation period definition",
        "question": "What is an Accumulation Period?",
        "expected_contains": ["Accumulation Period"]
    },
    {
        "name": "Copayment definition",
        "question": "What is a Copayment?",
        "expected_contains": ["Copayment"]
    },

    # ----------------------------
    # PRESCRIPTION COVERAGE TESTS
    # ----------------------------
    {
        "name": "Tier 1 prescription",
        "question": "What is the cost of a Tier 1 prescription drug?",
        "expected_contains": ["$15"]
    },
    {
        "name": "Generic 30-day supply",
        "question": "How much is a 30-day supply of a generic drug?",
        "expected_contains": ["30-day", "$"]
    },

    # ----------------------------
    # SERVICE COST TESTS
    # ----------------------------
    {
        "name": "Skilled nursing facility",
        "question": "What is the cost of skilled nursing facility care?",
        "expected_contains": ["$250"]
    },
    {
        "name": "Home health services",
        "question": "What does home health care cost?",
        "expected_contains": ["No charge"]
    },

    # ----------------------------
    # NEGATIVE TESTS (GROUNDING)
    # ----------------------------
    {
        "name": "CEO test",
        "question": "Who is the CEO of Kaiser?",
        "expected_exact": "I don't know."
    },
    {
        "name": "Stock price test",
        "question": "What is Kaiser’s stock price?",
        "expected_exact": "I don't know."
    },

    # ----------------------------
    # MULTI-CHUNK REASONING TEST
    # ----------------------------
    # {
    #     "name": "Costs that count toward maximum",
    #     "question": "What costs count toward the out-of-pocket maximum?",
    #     "expected_contains_any": ["Copayment", "Coinsurance"]
    # },
]


# ----------------------------
# Test runner
# ----------------------------

def run_tests():
    print("\nRunning RAG Test Cases...\n")
    passed = 0

    for test in TEST_CASES:
        print(f"TEST: {test['name']}")
        answer = agent.answer_query(test["question"])

        print("Question:", test["question"])
        print("Answer:", answer)
        print()

        if "expected_exact" in test:
            if answer.strip().lower() == test["expected_exact"].lower():
                print("✅ PASSED\n")
                passed += 1
            else:
                print("❌ FAILED")
                print("Expected exact:", test["expected_exact"])
                print("Actual:", answer)
                print()

        elif "expected_contains" in test:
            if text_matches(answer, test["expected_contains"]):
                print("✅ PASSED\n")
                passed += 1
            else:
                print("❌ FAILED")
                print("Expected to contain:", test["expected_contains"])
                print("Actual:", answer)
                print()

        elif "expected_contains_any" in test:
            if text_matches_any(answer, test["expected_contains_any"]):
                print("✅ PASSED\n")
                passed += 1
            else:
                print("❌ FAILED")
                print("Expected one of:", test["expected_contains_any"])
                print("Actual:", answer)
                print()

        print("-" * 60)

    print(f"\nFinal Score: {passed}/{len(TEST_CASES)} tests passed.\n")


if __name__ == "__main__":
    run_tests()