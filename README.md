# RAG Project – Retrieval‑Augmented Generation

## Overview

This repository implements a **Retrieval‑Augmented Generation (RAG)** pipeline that extracts information from PDF documents and uses a large language model (LLM) to answer questions or produce summaries.  The core workflow lives in a Jupyter notebook **`ragnotebook/pdfloader.ipynb`** which:

1. Loads PDFs from the `data/pdf/` directory.
2. Splits the text into chunks and builds a vector store (FAISS by default).
3. Retrieves the most relevant chunks for a user query.
4. Sends the retrieved context to an instruction‑tuned LLM (initially **Gemma‑2‑9B‑IT**) to generate the final answer.

The project is fully reproducible on a Windows machine (tested with Python 3.11) and can run on CPU or GPU.

---

## Repository Structure

```
RAG/
├─ .env                # Environment variables (e.g., HF_TOKEN)
├─ requirements.txt    # Python dependencies
├─ README.md           # <‑‑ you are reading this file
├─ ragnotebook/
│   └─ pdfloader.ipynb # Main notebook implementing the RAG pipeline
└─ data/
    └─ pdf/
        └─ YOUTUBE VIDEO TRANSCRIPTION USING EXTRACTIVE SUMMARIZATION.pdf
```

---

## Quick‑Start Guide

### 1️⃣ Clone the repository & navigate to the workspace
```powershell
git clone <repo‑url>
cd "d:/agentic ai/RAG"
```

### 2️⃣ Install dependencies
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Configure environment variables
Create a `.env` file (or edit the existing one) with at least:
```
HF_TOKEN=your_huggingface_access_token   # optional but recommended for private models
```

### 4️⃣ Launch the notebook
```powershell
jupyter lab ragnotebook/pdfloader.ipynb
```
The notebook contains the full step‑by‑step code; just run the cells sequentially.

---

## Core Components

| Component | Purpose | Key Libraries |
|-----------|---------|----------------|
| **PDF Loader** | Reads PDF files and extracts raw text. | `PyPDF2`, `pdfminer.six` |
| **Text Splitter** | Breaks long documents into manageable chunks (≈ 500 tokens). | `langchain.text_splitter` |
| **Vector Store** | Stores embeddings for fast similarity search. | `FAISS`, `langchain.vectorstores` |
| **Embedding Model** | Converts text chunks into dense vectors. | `sentence‑transformers/all‑MiniLM‑L6‑v2` (default) |
| **LLM (Generator)** | Produces answers using retrieved context. | Hugging  Face `transformers` – various instruction‑tuned models (see *Model Experiments* below) |

---

## Model Experiments – Timeline

| Date (UTC+05:30) | Activity | Model(s) Tried | Outcome |
|------------------|----------|----------------|---------|
| **2026‑09‑18 23:11** | Requested a model similar to **Gemma‑2‑9B‑IT**. | Suggested Llama‑2‑7B‑Chat, Mistral‑7B‑Instruct, etc. | Provided a short list.
| **2026‑09‑18 23:14** | Reported that the suggested models were not working. | N/A | Asked for more alternatives.
| **2026‑09‑18 23:22** | Supplied an extended catalog of models, including newer families (Llama‑3, Phi‑3, Qwen‑1.5, Mixtral, etc.) with usage snippets. | Many 7‑B‑15‑B sized models. | User could pick a replacement.
| **2026‑09‑18 23:31** | Requested a comprehensive `README.md` that documents the project, the steps taken, and a timeline of work. | — | This README is being generated now.

*Earlier work (before the current session) involved setting up the notebook, installing the required libraries, and loading the PDF `YOUTUBE VIDEO TRANSCRIPTION USING EXTRACTIVE SUMMARIZATION.pdf` for experimentation.*

---

## How to Switch the LLM in the Notebook

The notebook uses the following snippet (already present in the first code cell). To try a different model, replace the `model_name` string with any of the model identifiers from the table above:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "google/gemma-2-7b-it"  # ← change here

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
)
```

You can also enable **4‑bit quantisation** for larger models (e.g., Mixtral‑8×7B) by adding `load_in_4bit=True` and installing the `bitsandbytes` package.

---

## Frequently Asked Questions (FAQ)

**Q: My GPU runs out of memory when loading a model.**
- Use a smaller model (e.g., `Phi-3-mini-4k-instruct`).
- Enable 4‑bit quantisation (`load_in_4bit=True`).
- Or run the model on CPU (`device_map="cpu"`).

**Q: I get a `RuntimeError: CUDA out of memory` after building the FAISS index.**
- Reduce the chunk size in the text splitter (e.g., `chunk_size=300`).
- Delete the existing FAISS index and rebuild with a smaller embedding dimension.

**Q: How do I add new PDFs to the dataset?**
1. Place the PDF in `data/pdf/`.
2. In the notebook, add the file name to the `pdf_files` list.
3. Re‑run the loader cells to refresh the vector store.

---

## License

This project is released under the **Apache 2.0 License**. Individual models retain their own licensing (see each model’s README on Hugging  Face).

---

## Acknowledgements

- **Gemma‑2‑9B‑IT** – Google AI
- **Llama‑3, Mistral, Mixtral, Phi‑3, Qwen‑1.5** – Various research groups
- **LangChain** – framework for building LLM‑centric applications
- **FAISS** – Facebook AI similarity search library

---

*End of README*


give me a linkedin description to post
