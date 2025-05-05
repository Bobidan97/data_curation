import os
import glob
from typing import List, Dict, Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_documents(directory_path: str) -> List[Any]:
    """
    Grab all docs from a directory - currently handles txt and pdf files.

    Args:
        directory_path: Where your docs live

    Returns:
        List of loaded docs ready for processing
    """
    documents = []

    # Grab all text files - recursive flag lets us search subdirectories too
    for txt_path in glob.glob(os.path.join(directory_path, "**/*.txt"), recursive=True):
        loader = TextLoader(txt_path)
        documents.extend(loader.load())

    # Same deal for PDFs
    for pdf_path in glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True):
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    return documents


def chunk_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """
    Chop docs into bite-sized chunks for better retrieval.

    Args:
        documents: Docs to slice and dice
        chunk_size: How big each chunk should be (chars)
        chunk_overlap: Overlap between chunks to maintain context

    Returns:
        Chunked docs ready for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks: List[Any]) -> FAISS:
    """
    Turn chunks into vectors using FAISS - fast and efficient for similarity search.

    Args:
        chunks: Document chunks to vectorize

    Returns:
        FAISS vector store ready for querying
    """
    # Using MPNet - solid balance of quality and speed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def query_vector_store(vector_store: FAISS, query: str, top_k: int = 5) -> List[Any]:
    """
    Find the most relevant chunks for a query.

    Args:
        vector_store: Your vectorized documents
        query: What you're looking for
        top_k: How many results to grab

    Returns:
        The juicy chunks that best match your query
    """
    results = vector_store.similarity_search(query, k=top_k)
    return results


class RAGSystem:
    """
    One-stop shop for RAG - handles docs, chunking, and retrieval in one place.
    """

    def __init__(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Set up the RAG system.

        Args:
            directory_path: Where your docs live
            chunk_size: Size of chunks (bigger = more context, smaller = more precise)
            chunk_overlap: How much chunks should overlap (prevents context loss)
        """
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None

    def process_documents(self) -> None:
        """
        Pipeline to process docs: load → chunk → vectorize.
        This is the heavy lifting step - might take a while for large doc collections.
        """
        # Step 1: Load up the docs
        documents = load_documents(self.directory_path)

        # Step 2: Chunk 'em up
        chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)

        # Step 3: Create the vector store - this is where the magic happens
        self.vector_store = create_vector_store(chunks)

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Find the good stuff related to your query.

        Args:
            query_text: What you're asking about
            top_k: Number of chunks to return

        Returns:
            Dict with query, results, and the combined content
        """
        if self.vector_store is None:
            raise ValueError("You need to run process_documents() first! No vectors to search yet.")

        # Find the most similar chunks
        results = query_vector_store(self.vector_store, query_text, top_k)

        # Stitch together the content from all retrieved chunks
        content = "\n\n".join([doc.page_content for doc in results])

        return {
            "query": query_text,
            "results": results,
            "content": content
        }


def main():
    """
    Quick demo of how to use this thing.
    """
    # Fire up the RAG system
    rag = RAGSystem(directory_path="./documents")

    # Process the docs - this might take a while depending on doc size
    rag.process_documents()

    # Ask it something
    query = "Who is Thomas Hughes?"
    result = rag.query(query)

    # Check out what we got back
    print(f"Query: {result['query']}")
    print("\nRelevant Content:")
    print(result['content'])


if __name__ == "__main__":
    main()