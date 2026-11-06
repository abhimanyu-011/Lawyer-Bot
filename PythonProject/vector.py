import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_BASE = BASE_DIR / "chunks"
DB_PATH = BASE_DIR / "vectordb"

os.makedirs(DB_PATH, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
metadata = []

print("📂 Reading chunk files from", CHUNKS_BASE, "...")

for item in os.listdir(CHUNKS_BASE):
    item_path = os.path.join(CHUNKS_BASE, item)
    # Handle files directly in chunks/
    if os.path.isfile(item_path) and item.endswith(".txt"):
        with open(item_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            texts.append(content)
            metadata.append({
                "source_file": "custom", # fallback category for flat files
                "chunk_file": item
            })
    # Handle categorized subdirectories in chunks/
    elif os.path.isdir(item_path):
        for file in os.listdir(item_path):
            if file.endswith(".txt"):
                file_path = os.path.join(item_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    texts.append(content)
                    metadata.append({
                        "source_file": item,
                        "chunk_file": file
                    })

print(f"🔢 Total chunks: {len(texts)}")
if not texts:
    print("❌ No chunks found! Please add text files to the 'chunks' folder.")
    exit(1)

print("🧠 Creating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, str(DB_PATH / "lawyerbot.index"))

with open(DB_PATH / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("✅ Vector DB created successfully!")
print("📁 Saved in:", DB_PATH)
