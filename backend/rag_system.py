import os, json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

def load_ipynb_documents(directory: str) -> List[Document]:
    docs = []

    for filename in os.listdir(directory):
        if not filename.endswith(".ipynb"):
            continue

        path = os.path.join(directory, filename)

        with open(path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        current_heading = ""
        heading_level = 0
        section_text = ""
        section_cells = []

        for i, cell in enumerate(notebook.get("cells", [])):
            cell_type = cell.get("cell_type")
            source = "".join(cell.get("source", [])).strip()
            if not source:
                continue

            #detect headings in markdown cells
            if cell_type == "markdown":
                lines = source.splitlines()
                for line in lines:
                    if line.startswith("#"):
                        # flush previous section if it exists
                        if section_text:
                            docs.append(
                                Document(
                                    page_content=section_text,
                                    metadata={
                                        "source": filename,
                                        "heading": current_heading,
                                        "level": heading_level,
                                        "cell_index_range": (section_cells[0],
                                                             section_cells[-1]) if section_cells else None
                                    }
                                )
                            )
                            section_text = ""
                            section_cells = []

                        # new heading
                        heading_level = line.count("#")
                        current_heading = line.strip("# ").strip()
                        section_text += line + "\n"
                        section_cells.append(i)
                    else:
                        section_text += line + "\n"
                        section_cells.append(i)

            #include code cells
            elif cell_type == "code":
                section_text += f"```python\n{source}\n```\n"
                section_cells.append(i)

        # flush last section
        if section_text:
            docs.append(
                Document(
                    page_content=section_text,
                    metadata={
                        "source": filename,
                        "heading": current_heading,
                        "level": heading_level,
                        "cell_index_range": (section_cells[0], section_cells[-1]) if section_cells else None
                    }
                )
            )
    return docs


def chunk_sections(sections: List[Document], chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(sections)
    return chunks


def create_embeddings(chunks: List[Document]) -> Tuple[OpenAIEmbeddings, np.ndarray]:
    if not chunks:
        raise ValueError("❌ No document chunks to index.")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [doc.page_content for doc in chunks]
    embeddings_matrix = np.array(
        embedding_model.embed_documents(texts)
    )

    print(f"Created embeddings matrix with shape {embeddings_matrix.shape}")

    return embedding_model, embeddings_matrix


def query_vector_store(query: str, documents: List[Document], embedding_model,
                       embeddings_matrix: np.ndarray , top_k: int = 5) -> List[Document]:
    """
    Find the most relevant chunks for a query through cosine similarity.
    """
    query_embedding = np.array(
        embedding_model.embed_query(query)
    ).reshape(1, -1)

    #compute cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]

    #rank and select top-k
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [documents[i] for i in top_indices]


class RAGSystem:
    """
    One-stop shop for RAG - handles docs, chunking, and retrieval in one place.
    """

    def __init__(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Document] = []
        self.embeddings_matrix: np.ndarray | None = None
        self.embedding_model: OpenAIEmbeddings | None = None

    def process_documents(self) -> None:
        sections = load_ipynb_documents(self.directory_path)
        self.chunks = chunk_sections(sections, self.chunk_size, self.chunk_overlap)
        self.embedding_model, self.embeddings_matrix = create_embeddings(self.chunks)

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:

        if self.embeddings_matrix is None:
            raise ValueError("You need to run process_documents() first")

        results = query_vector_store(
            query = query_text,
            documents = self.chunks,
            embeddings_matrix = self.embeddings_matrix,
            embedding_model = self.embedding_model,
            top_k = top_k,
        )

        content = "\n\n".join([doc.page_content for doc in results])

        return {
            "query": query_text,
            "results": results,
            "content": content
        }
