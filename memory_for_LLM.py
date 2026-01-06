import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"


# -----------------------------
# Step 1: Load PDFs
# -----------------------------
def load_pdf_files(data_path: str):
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    return documents


documents = load_pdf_files(DATA_PATH)
print(f"Loaded {len(documents)} PDF pages")


# -----------------------------
# Step 2: Create Chunks
# -----------------------------
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


text_chunks = create_chunks(documents)
print(f"Created {len(text_chunks)} text chunks")


# -----------------------------
# Step 3: Embedding Model
# -----------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = get_embedding_model()


# -----------------------------
# Step 4: Store in FAISS
# -----------------------------
os.makedirs(DB_FAISS_PATH, exist_ok=True)

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("FAISS vector store saved successfully")
