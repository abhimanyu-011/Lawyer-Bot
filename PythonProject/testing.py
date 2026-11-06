import faiss
import json
import os
from sentence_transformers import SentenceTransformer

DB_PATH = r"C:\Users\sadik\OneDrive\Desktop\lawyerbot_dataset\vectordb"
CHUNKS_BASE = r"C:\Users\sadik\OneDrive\Desktop\lawyerbot_dataset\chunks"

index = faiss.read_index(os.path.join(DB_PATH, "lawyerbot.index"))

with open(os.path.join(DB_PATH, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

query = input("Ask a legal question: ")

query_vec = model.encode([query])
D, I = index.search(query_vec, 3)  # top 3 matches

print("\n🔍 Top matching legal chunks:\n")

for idx in I[0]:
    info = metadata[idx]
    chunk_folder = info["source_file"]
    chunk_file = info["chunk_file"]

    chunk_path = os.path.join(CHUNKS_BASE, chunk_folder, chunk_file)

    print(f"📂 Source: {chunk_folder} | 📄 Chunk: {chunk_file}")
    print("-" * 60)

    with open(chunk_path, "r", encoding="utf-8") as f:
        print(f.read()[:800])  # show first 800 characters
        print("\n")
