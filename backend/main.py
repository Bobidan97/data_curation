from llm import build_chembl_query_from_rag
from rag_system import RAGSystem
from chembl_query import get_molecules_activity_with_filters

def main():
    user_input = input("🔎 Enter your bioactivity query: ")

    #step 1: Get documentation from notebook (RAG)
    rag = RAGSystem(directory_path=r"C:\Users\Alex Bal\PycharmProjects\data_curation\documents")
    rag.process_documents()
    rag_result = rag.query(user_input)
    context_from_docs = rag_result['content']

    print("\n📚 Retrieved ChEMBL API usage context:")
    print(context_from_docs)

    #step 2: LLM generates a ChEMBL API query plan using user input + notebook content
    chembl_query_plan = build_chembl_query_from_rag(user_input, context_from_docs)

    print("\n🧠 Generated Query Plan:")
    print(chembl_query_plan)

    #step 3: Execute the interpreted ChEMBL query
    results = get_molecules_activity_with_filters(chembl_query_plan)

    print("\n💊 Results:")
    print(results.columns)
    print(results)

if __name__ == "__main__":
    main()
