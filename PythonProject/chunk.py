import os

INPUT_FOLDER = r"YOUR_INPUT_FILE"
OUTPUT_BASE = r"YOUR_OUTPUT_FILE"


CHUNK_SIZE = 500   # words per chunk
OVERLAP = 80       # overlapping words

os.makedirs(OUTPUT_BASE, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith("_clean.txt"):
        file_path = os.path.join(INPUT_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            words = f.read().split()

        out_folder = os.path.join(OUTPUT_BASE, filename.replace("_clean.txt", ""))
        os.makedirs(out_folder, exist_ok=True)

        start = 0
        chunk_id = 1

        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk_file = os.path.join(out_folder, f"chunk_{chunk_id:04d}.txt")
            with open(chunk_file, "w", encoding="utf-8") as out:
                out.write(chunk_text)

            chunk_id += 1
            start += CHUNK_SIZE - OVERLAP


        print(f"✅ Chunked: {filename} → {chunk_id-1} chunks")

print("\n🎯 All files chunked successfully!")
