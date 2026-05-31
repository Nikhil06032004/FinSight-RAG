<div align="center">

# 💹 FinSight-RAG

### AI-Powered Financial Regulatory Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green?style=for-the-badge)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-orange?style=for-the-badge)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-purple?style=for-the-badge)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A **production-grade Retrieval-Augmented Generation (RAG)** system for financial regulatory compliance — enabling analysts to query Basel III, IFRS 9, RBI, and ECB documents using natural language, with full source citation and zero hallucinations.

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Usage](#-usage) · [Project Structure](#-project-structure)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [Module Reference](#-module-reference)
- [Key Design Decisions](#-key-design-decisions)
- [Roadmap](#-roadmap)
- [Financial Disclaimer](#️-financial-disclaimer)

---

## 🔍 Overview

**FinSight-RAG** addresses a real operational gap in banking and financial services: analysts spend significant time manually searching through dense regulatory documents — Basel III frameworks, IFRS standards, RBI circulars, ECB supervision guides — to answer compliance and credit risk queries.

This system solves that by grounding every LLM response in a **curated corpus of financial regulatory documents**, preventing hallucination and ensuring every answer is traceable to a source. Built as a direct proof-of-concept for AI integration in financial risk and reporting workflows.

**What makes it unique for banking environments:**
- Region-aware retrieval — EU, APAC, GLOBAL metadata filtering mirrors multi-regional bank operations
- Query audit logging — every question, response, and source reference is persisted to SQLite for compliance traceability
- Deterministic outputs — temperature set to 0.1, enforcing consistent, professional-grade regulatory answers
- Strict prompt guardrails — LLM is instructed to never speculate or provide investment advice

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Regulatory Q&A** | Answers grounded in retrieved financial documents via RAG |
| 🌍 **Region-Aware Retrieval** | EU, APAC, GLOBAL metadata filtering across multi-regional documents |
| 📄 **Multi-Document Ingestion** | Loads and indexes 5 high-priority regulatory PDFs with metadata tagging |
| 🧠 **Semantic Search** | FAISS vector store for fast, accurate similarity retrieval |
| ⚡ **Ultra-Fast Inference** | Groq-hosted LLaMA 3.1 8B for sub-second response times |
| 📚 **Source Transparency** | Every answer includes collapsible source document references with page numbers |
| 🔒 **Hallucination Prevention** | Strict prompt engineering — answers only from context, never external knowledge |
| 🧾 **Audit Logging** | Every query logged to SQLite with timestamp, source, and response latency |
| 📥 **Chat Export** | Download full conversation history as `.txt` for compliance records |
| 🔑 **Flexible API Key Management** | Supports `.env` file or runtime UI input |
| 🗂️ **Structured Logging** | File + console logging with timestamps across all modules |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                       │
│                   Streamlit Web UI (Chat)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │  User Query
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                           │
│                                                             │
│   ┌──────────────┐     ┌────────────────────────────────┐  │
│   │   Retriever  │────▶│  ChatPromptTemplate            │  │
│   │  (FAISS k=5) │     │  (Finance Compliance Prompt)   │  │
│   │  Region Filter│    └───────────────┬────────────────┘  │
│   └──────┬───────┘                     │                    │
│          │  Top-K Chunks               │  Prompt + Context  │
│          ▼                             ▼                    │
│   ┌──────────────┐     ┌────────────────────────────────┐  │
│   │  FAISS Index │     │  ChatGroq (LLaMA 3.1-8B)       │  │
│   │  + Metadata  │     │  Temp: 0.1 | Tokens: 2048      │  │
│   │  (Region/    │     └───────────────┬────────────────┘  │
│   │   Source/    │                     │                    │
│   │   Topic tags)│                     │  StrOutputParser   │
│   └──────────────┘                     │                    │
└───────────────────────────────────────┼────────────────────┘
                                         ▼
                              Structured Regulatory Answer
                             + Source Doc + Region + Page No.
                                         │
                                         ▼
                              ┌──────────────────┐
                              │  SQLite Audit Log │
                              │  (query, source,  │
                              │   latency, region)│
                              └──────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                        │
│                                                             │
│  PDF Files ──▶ PyPDFLoader ──▶ RecursiveTextSplitter       │
│   (5 docs)      + Metadata       chunk=1000, overlap=150    │
│                  Tagging                  │                  │
│               (region/source/             ▼                  │
│                topic per doc) HuggingFaceEmbeddings          │
│                               (all-MiniLM-L6-v2, 384-dim)  │
│                                           │                  │
│                                           ▼                  │
│                              FAISS VectorStore.save_local    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **LLM** | Groq / LLaMA 3.1 | 8B Instant | Regulatory response generation |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 5.0.0 | Semantic text encoding |
| **Vector Store** | FAISS (CPU) | 1.11.0 | Similarity search over regulatory chunks |
| **Orchestration** | LangChain | 0.3.x | RAG pipeline management |
| **Web UI** | Streamlit | 1.46.x | Interactive frontend |
| **PDF Loader** | PyPDF + LangChain | 5.7.0 | Document ingestion with metadata |
| **Audit Store** | SQLite | built-in | Query logging and traceability |
| **Environment** | python-dotenv | 1.1.1 | Secret management |
| **Deep Learning** | PyTorch | 2.7.x | Embedding model backend |

---

## 📁 Project Structure

```
FinSight-RAG/
│
├── 📂 data/
│   └── docs/                         # Place regulatory PDFs here
│       ├── basel3.pdf
│       ├── ifrs9.pdf
│       ├── rbi_credit.pdf
│       ├── ecb_guide.pdf
│       └── bnp_2023.pdf
│
├── 📂 vectorstore/
│   └── db_faiss/                     # FAISS index (auto-generated)
│       ├── index.faiss
│       └── index.pkl
│
├── 📂 app/
│   ├── ingest.py                     # PDF ingestion & FAISS builder
│   ├── chain.py                      # RAG chain + prompt + classifier
│   └── logger.py                     # SQLite audit logger
│
├── 📂 logs/                          # Application log files
│   └── streamlit_rag.log
│
├── 📂 tests/
│   └── test_all.py                   # Unit tests — chain, logger, classifier
│
├── 📄 streamlit_app.py               # Main Streamlit web application
├── 📄 config.py                      # Centralized configuration
├── 📄 requirements.txt               # Dependencies
├── 📄 .env.example                   # API key template
├── 📄 .gitignore
└── 📄 README.md
```

---

## ✅ Prerequisites

- **Python** 3.9 or higher
- A free **Groq API key** — obtain one at [console.groq.com](https://console.groq.com)
- `pip` or `uv` package manager
- Regulatory PDF files in `data/docs/` before running ingestion

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Nikhil06032004/FinSight-RAG.git
cd FinSight-RAG
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** First install includes PyTorch (~2 GB) for the embedding model. Expect a few minutes.

### 4. Set Up API Key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 5. Add Regulatory Documents

Place PDFs in `data/docs/`. Recommended sources (all free):

| Document | Source |
|---|---|
| Basel III Framework | [bis.org](https://www.bis.org/bcbs/publ/d424.pdf) |
| IFRS 9 Standard | [ifrs.org](https://www.ifrs.org) |
| RBI Credit Risk Guidelines | [rbi.org.in](https://www.rbi.org.in) |
| ECB Banking Supervision Guide | [bankingsupervision.europa.eu](https://www.bankingsupervision.europa.eu) |
| BNP Paribas Annual Report 2023 | [invest.bnpparibas.com](https://invest.bnpparibas.com) |

### 6. Build the Vector Store

```bash
python app/ingest.py
```

Expected output:
```
[LOADING] Basel III Framework ... → 312 pages | 890 chunks
[LOADING] IFRS 9 Standard ...     → 184 pages | 510 chunks
[LOADING] RBI Credit Risk ...     → 96 pages  | 278 chunks
[LOADING] ECB Banking Guide ...   → 148 pages | 420 chunks
[LOADING] BNP Annual Report ...   → 256 pages | 730 chunks
[EMBEDDING] Total chunks: 2828 — this will take a few minutes...
[DONE] FAISS index saved to vectorstore/db_faiss
```

### 7. Launch the App

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Configuration

All parameters centralized in `config.py`:

```python
class Config:
    MODEL_NAME      = "llama-3.1-8b-instant"
    TEMPERATURE     = 0.1               # Low — deterministic regulatory answers
    MAX_TOKENS      = 2048
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DB_FAISS_PATH   = "vectorstore/db_faiss"
    TOP_K_RESULTS   = 5
    LOG_FILE        = "logs/streamlit_rag.log"
```

Ingestion chunk parameters in `app/ingest.py`:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Larger chunks — regulatory text needs paragraph context
    chunk_overlap=150    # Overlap preserves cross-paragraph continuity
)
```

---

## 💡 Usage

### Web Interface

1. Run `streamlit run streamlit_app.py`
2. Configure Groq API key in sidebar (or use `.env`)
3. Select region filter — ALL / EU / APAC / GLOBAL
4. Type your compliance query in the chat input
5. View structured answer with collapsible source references
6. Use **Export Chat** to download conversation for audit records

**Example queries:**
- "What is the minimum CET1 capital ratio under Basel III?"
- "How does IFRS 9 define credit impairment and expected credit loss?"
- "What are RBI guidelines on credit risk provisioning for APAC banks?"
- "What was BNP Paribas total credit risk exposure in 2023?"
- "Explain the Liquidity Coverage Ratio requirements under Basel III."

### Running Tests

```bash
pytest tests/test_all.py -v
```

---

## 📖 Module Reference

### `app/ingest.py` — Ingestion Pipeline

| Function | Description |
|---|---|
| `ingest_documents()` | Loads PDFs → tags metadata (region/source/topic) → chunks → embeds → saves FAISS index |

### `app/chain.py` — RAG Chain

| Function | Description |
|---|---|
| `create_rag_chain(llm, retriever)` | Builds full LangChain LCEL pipeline with finance compliance prompt |
| `classify_query(query)` | Classifies query into Credit Risk / Capital Adequacy / Compliance / Reporting / Liquidity |
| `get_confidence_label(scores)` | Converts FAISS L2 distances to High / Medium / Low confidence label |
| `load_chain(region_filter)` | Loads FAISS index + builds retriever with optional region metadata filter |

### `app/logger.py` — Audit Logger

| Function | Description |
|---|---|
| `init_db()` | Creates SQLite `query_log` table if not exists |
| `log_query(...)` | Persists query, type, region, response, sources, confidence, latency |
| `fetch_all_logs()` | Returns full query history as list of dicts |
| `fetch_stats()` | Returns aggregated stats — total queries, avg latency, by type/region/confidence |

### `streamlit_app.py` — Web Application

| Function | Description |
|---|---|
| `initialize_embeddings()` | `@st.cache_resource` — loads HuggingFace model once per session |
| `initialize_vectorstore()` | `@st.cache_resource` — loads FAISS index once per session |
| `initialize_llm()` | Creates `ChatGroq` instance from env or session-state API key |
| `create_rag_chain()` | Builds LangChain LCEL pipeline with finance prompt |
| `format_source_documents()` | Renders HTML collapsible source cards with region and page info |
| `render_sidebar()` | API config, system status, document list, export actions |
| `main()` | App entry point — wires all components, handles chat loop |

---

## 🎯 Key Design Decisions

**Why temperature 0.1 instead of 0.7?** Financial and regulatory answers must be deterministic and consistent. Higher temperatures introduce variation that is acceptable for general Q&A but unacceptable in compliance contexts where the same question should always yield the same answer.

**Why metadata tagging per document instead of merging PDFs?** Merging destroys source traceability. Each chunk retains its region (EU/APAC/GLOBAL), source document name, and topic tag — enabling filtered retrieval and source citation, both critical for a banking compliance tool.

**Why FAISS over a hosted vector DB?** FAISS is entirely local — no external service, no latency, no sensitive regulatory data leaving the machine. Important in financial environments where document confidentiality is a compliance requirement.

**Why SQLite for audit logging?** Banks require audit trails. Every query, response, source reference, and latency metric is persisted — reflecting real-world financial systems compliance requirements. Schema is designed to be Oracle PL/SQL compatible for enterprise portability.

**Why chunk size 1000 vs standard 500?** Regulatory documents (Basel III, IFRS 9) contain dense, inter-dependent paragraphs. Smaller chunks lose context. 1000-character chunks with 150-character overlap preserve paragraph continuity and improve retrieval accuracy on technical regulatory content.

---

## 🗺️ Roadmap

| Priority | Enhancement |
|---|---|
| 🔴 High | Add `ConversationBufferMemory` for multi-turn regulatory discussions |
| 🔴 High | Query Analytics dashboard — topic trends, peak usage, confidence distribution |
| 🟡 Medium | Docker support (`Dockerfile` + `docker-compose.yml`) |
| 🟡 Medium | FastAPI REST layer for programmatic access |
| 🟡 Medium | Automatic index rebuild when new PDFs are added |
| 🟢 Low | Support DOCX and HTML regulatory documents |
| 🟢 Low | RAGAS evaluation metrics for retrieval quality scoring |
| 🟢 Low | CI/CD pipeline with GitHub Actions |

---

## ⚠️ Financial Disclaimer

> **This application is for reference and educational purposes only.** It is not a substitute for professional financial, legal, or compliance advice. Regulatory interpretations should always be verified with qualified compliance officers and legal counsel. Never make financial or regulatory decisions solely based on outputs from this system.

---

<div align="center">

Built by [Nikhil](https://github.com/Nikhil06032004) · Powered by LangChain, Groq & FAISS

</div>