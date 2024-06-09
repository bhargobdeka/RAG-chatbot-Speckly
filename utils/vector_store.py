# vector_store.py
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_vector_store(docs, store_path: Optional[str] = None) -> FAISS:
    """
    Creates a FAISS vector store from a list of documents.

    Args:
        docs (List[Document]): A list of Document objects containing the content to be stored.
        store_path (Optional[str]): The path to store the vector store locally. If None, the vector store will not be stored.

    Returns:
        FAISS: The FAISS vector store containing the documents.
    """
    # Creating text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    texts = text_splitter.split_documents(docs)

    # Embedding object
    embedding_model = OpenAIEmbeddings()

    # Create the FAISS vector store
    store = FAISS.from_documents(texts, embedding_model)

    # Save the vector store locally if a path is provided
    if store_path:
        store.save_local(store_path)

    return store

def get_local_store(store_path: str) -> FAISS:
    """
    Loads a locally stored FAISS vector store.

    Args:
        store_path (str): The path where the FAISS vector store is stored locally.

    Returns:
        FAISS: The loaded FAISS vector store.
    """
    # Load the embedding model
    embedding_model = OpenAIEmbeddings()
    
    # Load the vector store from the local path
    store = FAISS.load_local(store_path, embedding_model)

    return store