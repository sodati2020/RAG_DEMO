from app.policy_agent import PolicyAgent

agent = PolicyAgent(
    [
        "data/kaiser_1.pdf",
        "data/Kaiser_2.pdf",
    ]
)

while True:
    question = input("Ask policy question: ")
    print(agent.answer_query(question))
