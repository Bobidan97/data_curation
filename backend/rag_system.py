import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import NotebookLoader
from langchain_huggingface import HuggingFaceEmbeddings

def load_ipynb_documents(directory: str):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".ipynb"):
            path = os.path.join(directory, filename)
            loader = NotebookLoader(path)
            docs.extend(loader.load())
    return docs

def load_documents(directory: str):
    docs = []
    docs.extend(load_ipynb_documents(directory))
    print(f"✅ Loaded {len(docs)} documents.")
    return docs
def chunk_documents(docs: List[Any], chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"✅ Number of document chunks: {len(chunks)}")
    return chunks


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if not chunks:
        raise ValueError("❌ No document chunks to index.")
    return FAISS.from_documents(chunks, embeddings)


def query_vector_store(vector_store: FAISS, query: str, top_k: int = 5) -> List[Any]:
    """
    Find the most relevant chunks for a query.
    """
    return vector_store.similarity_search(query, k=top_k)


class RAGSystem:
    """
    One-stop shop for RAG - handles docs, chunking, and retrieval in one place.
    """

    def __init__(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None

    def process_documents(self) -> None:
        documents = load_documents(self.directory_path)
        chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        self.vector_store = create_vector_store(chunks)

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        if self.vector_store is None:
            raise ValueError("You need to run process_documents() first! No vectors to search yet.")
        results = query_vector_store(self.vector_store, query_text, top_k)
        content = "\n\n".join([doc.page_content for doc in results])
        return {
            "query": query_text,
            "results": results,
            "content": content
        }


def main():
    rag = RAGSystem(directory_path=r"C:\Users\Alex Bal\PycharmProjects\data_curation\documents")
    rag.process_documents()

    query = "Show me an example for collecting activity data"
    result = rag.query(query)

    print(f"\n🔍 Query: {result['query']}")
    print("\n📄 Relevant Content:\n")
    print(result['content'])


if __name__ == "__main__":
    main()
# def load_documents(directory_path: str) -> List[Document]:
#     """
#     Load Jupyter notebooks (.ipynb) from the directory.
#
#     Args:
#         directory_path: Path to directory containing .ipynb files
#
#     Returns:
#         List of LangChain Documents
#     """
#
#     documents = []
#     for nb_path in glob.glob(os.path.join(directory_path, "**/*.ipynb"), recursive=True):
#         with open(nb_path, "r", encoding="utf-8") as f:
#             notebook = json.load(f)
#         cells = notebook.get("cells", [])
#         for i, cell in enumerate(cells):
#             cell_type = cell.get("cell_type")
#             source = "".join(cell.get("source", []))
#             if cell_type in ["markdown", "code"] and source.strip():
#                 documents.append(
#                     Document(
#                         page_content=source.strip(),
#                         metadata={
#                             "source": os.path.basename(nb_path),
#                             "cell_index": i,
#                             "cell_type": cell_type
#                         }
#                     )
#                 )
#     return documents


# def chunk_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
#     """
#     Chop docs into bite-sized chunks for better retrieval.
#
#     Args:
#         documents: Docs to slice and dice
#         chunk_size: How big each chunk should be (chars)
#         chunk_overlap: Overlap between chunks to maintain context
#
#     Returns:
#         Chunked docs ready for embedding
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#     )
#
#     chunks = text_splitter.split_documents(documents)
#
#     return chunks
#
#
# def create_vector_store(chunks: List[Any]) -> FAISS:
#     """
#     Turn chunks into vectors using FAISS - fast and efficient for similarity search.
#
#     Args:
#         chunks: Document chunks to vectorize
#
#     Returns:
#         FAISS vector store ready for querying
#     """
#     # Using MPNet - solid balance of quality and speed
#     print(f"Number of document chunks: {len(chunks)}")
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     vector_store = FAISS.from_documents(chunks, embeddings)
#
#     return vector_store
#
#
# def query_vector_store(vector_store: FAISS, query: str, top_k: int = 5) -> List[Any]:
#     """
#     Find the most relevant chunks for a query.
#
#     Args:
#         vector_store: Your vectorized documents
#         query: What you're looking for
#         top_k: How many results to grab
#
#     Returns:
#         The juicy chunks that best match your query
#     """
#     results = vector_store.similarity_search(query, k=top_k)
#     return results
#
#
# class RAGSystem:
#     """
#     One-stop shop for RAG - handles docs, chunking, and retrieval in one place.
#     """
#
#     def __init__(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
#         """
#         Set up the RAG system.
#
#         Args:
#             directory_path: Where your docs live
#             chunk_size: Size of chunks (bigger = more context, smaller = more precise)
#             chunk_overlap: How much chunks should overlap (prevents context loss)
#         """
#         self.directory_path = directory_path
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.vector_store = None
#
#     def process_documents(self) -> None:
#         """
#         Pipeline to process docs: load → chunk → vectorize.
#         This is the heavy lifting step - might take a while for large doc collections.
#         """
#         # Step 1: Load up the docs
#         documents = load_documents(self.directory_path)
#
#         # Step 2: Chunk 'em up
#         chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
#
#         # Step 3: Create the vector store - this is where the magic happens
#         self.vector_store = create_vector_store(chunks)
#
#     def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
#         """
#         Find the good stuff related to your query.
#
#         Args:
#             query_text: What you're asking about
#             top_k: Number of chunks to return
#
#         Returns:
#             Dict with query, results, and the combined content
#         """
#         if self.vector_store is None:
#             raise ValueError("You need to run process_documents() first! No vectors to search yet.")
#
#         # Find the most similar chunks
#         results = query_vector_store(self.vector_store, query_text, top_k)
#
#         # Stitch together the content from all retrieved chunks
#         content = "\n\n".join([doc.page_content for doc in results])
#
#         return {
#             "query": query_text,
#             "results": results,
#             "content": content
#         }
#
#
# def main():
#     """
#     Quick demo of how to use this thing.
#     """
#     # Fire up the RAG system
#     rag = RAGSystem(directory_path="./documents")
#
#     # Process the docs - this might take a while depending on doc size
#     rag.process_documents()
#
#     # Ask it something
#     query = "Show me bioactivty data for cyclooxygenase"
#     result = rag.query(query)
#
#     # Check out what we got back
#     print(f"Query: {result['query']}")
#     print("\nRelevant Content:")
#     print(result['content'])
#
#
# if __name__ == "__main__":
#     main()
