import pdfplumber
import os

PDF_PATH = r"YOUR_RAW_PDF_PATH"

OUT_DIR = r"YOUR_DIR_PATH"
OUT_PATH = os.path.join(OUT_DIR, "NAME_OF_YOUR_FILE")

os.makedirs(OUT_DIR, exist_ok=True)

all_text = []

with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))

print("✅ Extracted text saved to:", OUT_PATH)
