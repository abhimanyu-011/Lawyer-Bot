# LawyerBot — AI Legal Assistant using RAG

An AI-powered legal chatbot for Kerala & Indian law, built using Retrieval Augmented Generation (RAG).
It retrieves relevant sections from legal documents and generates accurate, context-aware answers using Llama 3.3 70B via the Groq API.

## Features
- **Law Categories** — Civil, Criminal, Motor Vehicles, Constitution, Cyber IT Law, Custom Integration.
- **RAG Pipeline** — FAISS vector search retrieves exact law sections before generating answers.
- **Ultra-fast responses** — Powered by Groq API.
- **Local Web UI** — Built-in modern web interface.

## Local Windows Setup

1. **Clone the repository** (if not already done)
   ```bash
   git clone <repo-url>
   cd LawyerAI-legal-assistant-using-RAG-main
   ```

2. **Set up the Virtual Environment**
   ```bash
   python -m venv PythonProject\.venv
   PythonProject\.venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r PythonProject\requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the `PythonProject` directory (use `.env.example` as a template):
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Generate the FAISS Vector Database**
   (If the database is missing or you added new chunks, you must build the index)
   ```bash
   cd PythonProject
   python vector.py
   ```

6. **Run the Application**
   Start the FastAPI server:
   ```bash
   uvicorn api:app --reload
   ```

7. **Access the Chat Interface**
   Open your browser and navigate to:
   http://127.0.0.1:8000/

## Project Structure
- `PythonProject/api.py` — Main FastAPI backend
- `PythonProject/vector.py` — FAISS index builder
- `PythonProject/lawyerbot.html` — Chat UI (Served directly from `/`)
- `chunks/` — Processed text files used to build the FAISS index
- `vectordb/` — Contains the FAISS `.index` and metadata

## License
MIT License
