"""
LawyerBot — api.py
============================
College    : xxxxx
University : APJ Abdul Kalam Technological University
Developers : xxxxx
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq

# ══════════════════════════════════════════
#  PATHS — dynamically resolved to project root
# ══════════════════════════════════════════
BASE_DIR     = Path(__file__).resolve().parent.parent
VECTOR_DB    = BASE_DIR / "vectordb" / "lawyerbot.index"
METADATA     = BASE_DIR / "vectordb" / "metadata.json"
CHUNKS_DIR   = BASE_DIR / "chunks"
FRONTEND_HTML= Path(__file__).resolve().parent / "lawyerbot.html"
ENV_FILE     = Path(__file__).resolve().parent / ".env"

TOP_K = 5

# ══════════════════════════════════════════
#  LOAD ENV & CLIENTS
# ══════════════════════════════════════════
# LOAD ENV & CLIENTS
load_dotenv(ENV_FILE, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found in PythonProject/.env"
    )

groq_client = Groq(api_key=GROQ_API_KEY)
# ══════════════════════════════════════════
#  LOAD MODELS & VECTOR DB AT STARTUP
# ══════════════════════════════════════════
print("\nLoading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded")

print("Loading FAISS index...")
if not VECTOR_DB.exists():
    print(f"WARNING: FAISS index not found at: {VECTOR_DB}. Please run vector.py to generate it.")
    dim = embedder.get_sentence_embedding_dimension()
    faiss_index = faiss.IndexFlatL2(dim)
else:
    faiss_index = faiss.read_index(str(VECTOR_DB))
    print(f"FAISS index loaded — {faiss_index.ntotal} vectors")

print("Loading metadata...")
metadata = []
if METADATA.exists():
    with open(METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Metadata loaded — {len(metadata)} chunks")
else:
    print(f"WARNING: Metadata not found at: {METADATA}")

CATEGORY_SOURCES = {
    "civil":        ["civil", "civil_procedure", "kerala_civil", "property", "rent", "land"],
    "criminal":     ["criminal", "ipc", "bns", "bnss", "crpc", "kerala_police", "penal"],
    "motor":        ["motor", "vehicle", "mvact", "mva", "transport"],
    "constitution": ["constitution", "fundamental", "constitutional", "directive"],
    "cyber":        ["cyber", "it_act", "information_technology", "digital", "dpdp"],
    "custom":       ["custom", "university", "college", "organisation", "rules"],
}

def get_sources_for_category(category: str) -> List[str]:
    return CATEGORY_SOURCES.get(category.lower(), [])

def read_chunk_text(source_file: str, chunk_file: str) -> str:
    # First check in a source folder if it exists
    chunk_path_folder = CHUNKS_DIR / source_file / chunk_file
    # Fallback to flat directory in chunks/
    chunk_path_flat = CHUNKS_DIR / chunk_file
    
    if chunk_path_folder.exists():
        with open(chunk_path_folder, "r", encoding="utf-8") as f:
            return f.read().strip()
    elif chunk_path_flat.exists():
        with open(chunk_path_flat, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def retrieve_chunks(question: str, category: str, top_k: int = TOP_K) -> List[dict]:
    if faiss_index.ntotal == 0:
        return []

    question_embedding = embedder.encode([question], convert_to_numpy=True).astype("float32")
    search_k = top_k * 6
    # Ensure search_k doesn't exceed total elements
    search_k = min(search_k, faiss_index.ntotal)
    
    if search_k == 0:
        return []

    distances, indices = faiss_index.search(question_embedding, search_k)
    category_keywords = get_sources_for_category(category)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        source_file = meta.get("source_file", "").lower()
        chunk_file  = meta.get("chunk_file", "")

        if category_keywords:
            if not any(kw in source_file for kw in category_keywords):
                continue

        text = read_chunk_text(source_file, chunk_file)
        if not text:
            continue

        results.append({
            "text":        text,
            "source_file": source_file,
            "chunk_file":  chunk_file,
            "distance":    float(dist)
        })

        if len(results) >= top_k:
            break

    if not results:
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
            meta        = metadata[idx]
            source_file = meta.get("source_file", "")
            chunk_file  = meta.get("chunk_file", "")
            text        = read_chunk_text(source_file, chunk_file)
            if text:
                results.append({
                    "text":        text,
                    "source_file": source_file,
                    "chunk_file":  chunk_file,
                    "distance":    float(dist)
                })
            if len(results) >= top_k:
                break

    return results

BASE_INSTRUCTION = """You are LawyerBot, an expert AI legal assistant for Kerala, India.
You answer questions using retrieved legal document excerpts provided below.
Always cite the source document and relevant section/act in your answer.
Be clear, accurate, and easy for a non-lawyer to understand.
End every response with:
"WARNING: This is general legal information only. Please consult a licensed advocate for advice specific to your situation." """

SYSTEM_PROMPTS = {
    "civil": BASE_INSTRUCTION + "\\nYou specialize in Kerala Civil Law: property, land registration, tenancy, rent control, civil procedures, succession, Kerala Land Reforms Act, CPC, Limitation Act.",
    "criminal": BASE_INSTRUCTION + "\\nYou specialize in Kerala Criminal Law: FIR, bail, arrest, IPC/BNS offences, BNSS procedures, Kerala Police Act, trial process, anticipatory bail.",
    "motor": BASE_INSTRUCTION + "\\nYou specialize in Motor Vehicle Laws for Kerala: MVA 1988, Kerala MVD rules, fines, licensing, vehicle registration, insurance, MACT claims.",
    "constitution": BASE_INSTRUCTION + "\\nYou specialize in the Indian Constitution: Fundamental Rights, Directive Principles, writ petitions, PILs, landmark SC judgments, Articles 14, 19, 21, 32, 226.",
    "cyber": BASE_INSTRUCTION + "\\nYou specialize in Cyber and IT Law in India: IT Act 2000, IT Amendment 2008, DPDP Act 2023, cybercrime, online fraud, data privacy, Kerala Cyber Cell.",
    "custom": BASE_INSTRUCTION + "\\nYou handle custom institutional queries. Answer ONLY from the provided documents. If not found, clearly say: 'This information is not in the provided documents.'",
    "general": BASE_INSTRUCTION + "\\nAnswer general Indian and Kerala legal questions across all areas of law.",
}

def generate_answer(question: str, chunks: List[dict], category: str, chat_history: List[dict]) -> str:
    global groq_client
    if not groq_client:
        load_dotenv(str(ENV_FILE), override=True)
        key = os.getenv("GROQ_API_KEY")
        if key and key.strip() and key != "your_groq_api_key_here":
            groq_client = Groq(api_key=key)
        else:
            raise RuntimeError("GROQ_API_KEY is not configured.")

    if chunks:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Excerpt {i} — Source: {chunk['source_file']}]\\n{chunk['text']}"
            )
        context_block = "\\n\\n".join(context_parts)
        context_section = f"""
RETRIEVED LEGAL EXCERPTS
{context_block}
END OF EXCERPTS

Base your answer on the above excerpts. Cite sources where relevant.
"""
    else:
        context_section = "\\nNo specific documents retrieved. Answer from general knowledge of Indian and Kerala law.\\n"

    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["general"]) + context_section

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content

app = FastAPI(
    title="LawyerBot API",
    description="Kerala Legal AI — FAISS + Groq Llama RAG Pipeline",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    category: str
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str
    category: str
    sources: List[str]
    chunks_retrieved: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    category = request.category.lower()

    if category not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown category '{category}'. Valid: civil, criminal, motor, constitution, cyber, custom")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    user_question = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    if not user_question.strip():
        raise HTTPException(status_code=400, detail="No user question found.")

    chunks = retrieve_chunks(user_question, category)
    sources = list(set(c["source_file"] for c in chunks))

    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    try:
        reply = generate_answer(user_question, chunks, category, history)
    except Exception as e:
        error_msg = str(e)
        if "GROQ_API_KEY is not configured" in error_msg:
            raise HTTPException(status_code=401, detail="Groq API Key is missing. Please add it to the .env file in PythonProject.")
        elif "401" in error_msg or "auth" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid Groq API key. Check your .env file.")
        elif "429" in error_msg:
            raise HTTPException(status_code=429, detail="Groq rate limit reached. Wait a moment.")
        else:
            raise HTTPException(status_code=500, detail=f"LLM error: {error_msg}")

    return ChatResponse(
        reply=reply,
        category=category,
        sources=sources,
        chunks_retrieved=len(chunks)
    )

@app.get("/")
def root():
    if FRONTEND_HTML.exists():
        return FileResponse(FRONTEND_HTML)
    return {"error": "lawyerbot.html not found"}

@app.get("/api/status")
def api_status():
    return {
        "app":     "LawyerBot API",
        "version": "2.0.0",
        "status":  "running",
        "vectors": faiss_index.ntotal if faiss_index else 0,
        "chunks":  len(metadata),
        "tip":     "Visit /docs for interactive API testing"
    }

@app.get("/health")
def health():
    chunk_folders = []
    if CHUNKS_DIR.exists():
        chunk_folders = [f.name for f in CHUNKS_DIR.iterdir() if f.is_dir()]
    coverage = {}
    for cat, keywords in CATEGORY_SOURCES.items():
        matched = [f for f in chunk_folders if any(kw in f.lower() for kw in keywords)]
        coverage[cat] = matched if matched else "No matching folders found"

    return {
        "status":         "ok",
        "faiss_index":    str(VECTOR_DB),
        "vectors_loaded": faiss_index.ntotal if faiss_index else 0,
        "chunks_loaded":  len(metadata),
        "embed_model":    "all-MiniLM-L6-v2",
        "llm_model":      "llama-3.3-70b-versatile",
        "chunk_folders":  chunk_folders,
        "category_coverage": coverage
    }

@app.get("/sources")
def list_sources():
    sources = sorted(set(m.get("source_file", "unknown") for m in metadata))
    return {
        "total_sources": len(sources),
        "total_chunks":  len(metadata),
        "sources": sources
    }
