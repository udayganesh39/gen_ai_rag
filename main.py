import os
from pathlib import Path
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

SUPPORTED_TYPES = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".docx": "docx"
}

DATA_DIR = "data"
INDEX_BASE_DIR = "indexes"

def get_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_gemini_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

def load_docs(file_type, file_paths):
    docs = []
    for file_path in file_paths:
        if file_type == "pdf":
            docs.extend(PyPDFLoader(str(file_path)).load())
        elif file_type == "txt":
            docs.extend(TextLoader(str(file_path)).load())
        elif file_type == "docx":
            docs.extend(UnstructuredWordDocumentLoader(str(file_path)).load())
    return docs

def build_or_load_faiss(file_type, file_paths):
    embeddings = get_gemini_embeddings()
    index_dir = os.path.join(INDEX_BASE_DIR, f"{file_type}_index")

    if os.path.exists(index_dir):
        print(f"Loading existing FAISS index for {file_type}")
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Creating FAISS index for {file_type}")
        docs = load_docs(file_type, file_paths)
        faiss_index = FAISS.from_documents(docs, embeddings)
        faiss_index.save_local(index_dir)
        return faiss_index

def retrieve_all_sources(query, stores, k=4):
    all_docs = []
    for store in stores.values():
        all_docs.extend(store.similarity_search(query, k=k))
    return all_docs

def scan_data_directory():
    file_map = defaultdict(list)
    for path in Path(DATA_DIR).glob("*"):
        ext = path.suffix.lower()
        if ext in SUPPORTED_TYPES:
            file_type = SUPPORTED_TYPES[ext]
            file_map[file_type].append(path)
    return file_map

def main():
    os.makedirs(INDEX_BASE_DIR, exist_ok=True)

    # Detect all files and group by type
    file_groups = scan_data_directory()

    # Create/load vectorstores for each type
    stores = {
        file_type: build_or_load_faiss(file_type, file_paths)
        for file_type, file_paths in file_groups.items()
    }

    if not stores:
        print("No supported files found in 'data/' folder.")
        return

    query = input("Ask your question: ")
    context_docs = retrieve_all_sources(query, stores)
    context_text = "\n".join([doc.page_content for doc in context_docs])

    llm = get_gemini_llm()
    response = llm.invoke(f"Answer the following based on the context:\n{context_text}\n\nQuestion: {query}")
    print("\nResponse:\n", response.content)

if __name__ == "__main__":
    main()