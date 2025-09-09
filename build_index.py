# build_index.py
from modules.rag import build_index
import os

if __name__ == "__main__":
    if not os.path.exists("faiss_index"):
        print("FAISS index not found. Building index...")
        build_index()
        print("Index built successfully.")
    else:
        print("FAISS index already exists.")
