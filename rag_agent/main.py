from config import *
from tool_registry import *
from tools.document_tools import doc_index
from agent import agent_executor

def main():
    # First, index the documents
    print("--- Indexing Documents ---")
    index_result = doc_index("documents/Albert_Malvino.txt,documents/deep-learning-in-python-prerequisites.txt")
    print(index_result)

    # Example queries
    queries = [
        "List experience of Neha",
        "What's the paper 1605.08386 about?",
        "Who was the british king who established the church of England and in what year?",
        "Who was the british king who established the church of England and what was the role of his marriage in his decision?",
        "Who is the person identified as the shooter of Charlie Kirk?",
        "How much tariff has Trump levied on India?",
        "Tell me a programming joke",
        "What are the latest updates from NASA on the possibility of life on Mars?",
        "What is machine learning according to the deep learning prerequisites document?",
        "Explain what a diode is based on the electronics book"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n--- Response {i}: {query} ---")
        try:
            response = agent_executor.invoke({"input": query})
            print(response["output"])
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
