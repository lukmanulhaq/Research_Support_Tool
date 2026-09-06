import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None


def initialize_rag():
    global vector_store
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()

    if not docs:
        print("Warning: No PDFs found in 'data/' directory. Tool will return empty results.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="data/vector_db")


@tool
def extract_academic_evidence(query: str) -> str:
    """Use this tool to query the vector database for methodologies, metrics, and dataset sizes from the research papers."""
    if vector_store is None:
        return "Vector store not initialized. Please add PDFs to the data folder."

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)

    if not results:
        return "No relevant context found in the academic corpus."

    return "\n\n".join([doc.page_content for doc in results])
