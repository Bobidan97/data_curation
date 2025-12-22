import os, json
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import NotebookLoader
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

def load_ipynb_documents(directory: str) -> List[Document]:
    docs = []

    for filename in os.listdir(directory):
        if not filename.endswith(".ipynb"):
            continue

        path = os.path.join(directory, filename)

        with open(path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        for i, cell in enumerate(notebook.get("cells", [])):
            cell_type = cell.get("cell_type")
            source = "".join(cell.get("source", [])).strip()

            if not source:
                continue

            if cell_type == "code":
                source = f"```python\n{source}\n```"

            docs.append(
                Document(
                    page_content=source,
                    metadata={
                        "source": filename,
                        "cell_index": i,
                        "cell_type": cell_type
                    },
                )
            )

    print(f"✅ Loaded {len(docs)} documents from {directory}")
    return docs

def chunk_documents(docs: List[Any], chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"✅ Number of document chunks: {len(chunks)}")
    return chunks


def create_vector_store(chunks):
    if not chunks:
        raise ValueError("❌ No document chunks to index.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def query_vector_store(vector_store: FAISS, query: str, top_k: int = 5) -> List[Any]:
    """
    Find the most relevant chunks for a query.
    """
    results = vector_store.similarity_search(query, k = top_k)
    return results


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
        documents = load_ipynb_documents(self.directory_path)
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
