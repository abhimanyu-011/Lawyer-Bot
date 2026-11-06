import os
import re

INPUT_FOLDER = r"YOUR_INPUT_FOLDER"
OUTPUT_FOLDER = r"YOUR_INPUT_FOLDER"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".txt"):
        in_path = os.path.join(INPUT_FOLDER, filename)
        out_path = os.path.join(OUTPUT_FOLDER, filename.replace(".txt", "_clean.txt"))

        with open(in_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Clean text
        text = re.sub(r"\s+", " ", text)                 # remove extra spaces
        text = re.sub(r"\b\d{1,4}\b", "", text)          # remove page numbers
        text = re.sub(r"[^\w\s\.,;:()\-]", "", text)     # remove strange symbols

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text.strip())

        print(f"✅ Cleaned: {filename} → {os.path.basename(out_path)}")

print("\n🎯 All files cleaned successfully!")
