# 🔬 Contextual RAG Semantic Audit System

## 🚀 Project Overview

This is an AI-powered document comparison tool designed for **Audit & Compliance**. It uses **RAG (Retrieval-Augmented Generation)** to compare two documents (e.g., 2023 vs 2024 Annual Reports) and detect semantic discrepancies, not just text differences.

| Feature | Description |
| :--- | :--- |
| **Hybrid Extraction** | Uses **Microsoft MarkItDown** for Excel/Word and **IBM Docling** for complex PDFs. |
| **Dual AI Core** | Supports **Cloud APIs** (Gemini 2.5 / Claude 3.5) and **Local Models** (Ollama Llama 3). |
| **Smart Audit** | Two specialized modes: **Error Detection** (Translations) and **Variance Analysis** (Financials). |
| **Agentic Brain** | **Phase 4**: LangGraph-powered intelligent workflow orchestration (NEW). |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com) (Optional, for local privacy)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys (Cloud Only)

Create a `.env` file or enter keys in the sidebar:
```env
GEMINI_API_KEY=your_key_here
# or
CLAUDE_API_KEY=your_key_here
```

**API Key Sources:**
- **Claude:** https://console.anthropic.com/
- **Gemini:** https://ai.google.dev/

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
v2_agentic_dev/
├── app.py                      # Main Streamlit Interface (Run this!)
├── model_config.py             # LLM Factory (Cloud + Local)
├── document_upload_rag.py      # Document Processing & Vector Store
├── document_comparison_rag.py  # Semantic Audit & Risk Detection
├── agent_brain.py              # 🧠 Phase 4: Agentic Workflow (NEW)
├── requirements.txt            # Dependencies (with LangGraph)
├── .env                        # API Keys (DO NOT COMMIT)
├── docs/                       # Documentation
│   ├── OLLAMA_SETUP_GUIDE.md   # Local Ollama Setup
│   ├── COMPARISON.md           # Technical Comparison Details
│   ├── Quickstart.md           # Quick Reference
│   └── summary.md              # System Summary
├── test_report/                # Sample Test Documents
└── faiss_index/                # Vector Database (Auto-generated)
```

---

## 🎯 The Two Audit Modes (Critical!)

The application offers **two distinct audit modes** that solve different problems. **Understanding the difference is critical for using the tool correctly.**

### Mode 1: Error Detection (Default)

**Best for:** Comparing a **Source** vs. a **Translation/Draft**

**Examples:**
- English Contract vs. Chinese Contract
- Original Document vs. Translated Version
- Source PDF vs. Proofread Draft

**Logic:**
*"If Document A says 'Apple' and Document B says 'Banana', that is a **MISTAKE** that must be flagged."*

**Behavior:**
- Ignores formatting differences and minor rewording
- Flags **any meaning mismatch** between the two documents
- Reports logic errors, mistranslations, and missing critical information
- Uses stricter validation to catch errors that could cause problems

**Use Case Example:**
You have a Chinese contract that was translated from English. You want to ensure the translation is accurate and doesn't lose or distort any critical information.

---

### Mode 2: Variance Analysis (Year-over-Year)

**Best for:** Comparing **2023 Financial Report** vs. **2024 Financial Report**

**Examples:**
- Annual Report (2023) vs. Annual Report (2024)
- Q1 Report vs. Q2 Report
- Last Year's Data vs. This Year's Data

**Logic:**
*"If Revenue changed from $100M to $120M, that is a **VALID UPDATE**, but I **need to know about it** because it's a significant business change."*

**Behavior:**
- Uses a "Smart Filter" to ignore fluff and rewording
- **Highlights changes in:**
  - Numbers (revenue, profit, percentages, quantities)
  - Dates and time periods
  - Entity names (companies, people, stock codes)
  - Meaning reversals (e.g., "Outlook is positive" → "Outlook is negative")
- Ignores cosmetic changes like:
  - Rewording that means the same thing ("We grew 5%" vs. "Growth was 5%")
  - Formatting changes, headers, footers
  - General marketing copy rewrites

**Use Case Example:**
You're comparing a company's 2023 annual report with their 2024 annual report. You want to know what changed materially (revenue, headcount, leadership), but you don't care if they rewrote the marketing description.

---

## 🔑 Key Differences at a Glance

| Aspect | Error Detection | Variance Analysis |
|--------|-----------------|-------------------|
| **Purpose** | Find errors/mistakes | Find business changes |
| **Best for** | Translations, Proofreading | Annual Reports, YoY Comparisons |
| **What it flags** | All meaning differences | Only material/numeric changes |
| **Noise filtering** | Low (strict) | High (smart) |
| **Example finding** | "Translation says 'Red' but should be 'Blue'" | "Revenue: $100M → $120M (20% increase)" |

---

## 💻 Application Tabs

### Tab 1: Upload Documents
- Upload your two documents (PDF or text)
- System generates summaries and chunks them for comparison
- Creates searchable vector stores for retrieval
- **Tip:** Use descriptive names (e.g., "2023_Annual_Report.pdf", "2024_Annual_Report.pdf")

### Tab 2: Vector Compare
- Shows status of vector stores
- Quick vector-based comparison (FREE - no LLM calls)
- Line-by-line text differences
- (Future: Visual comparison of retrieved chunks)

### Tab 3: Semantic Audit
- Select your audit mode (Error Detection or Variance Analysis)
- Click "Run Audit"
- View results with expandable details on each finding
- Export results as JSON

---

## 📊 Results Format

Each finding (Risk Item) includes:

- **Risk Level:** CRITICAL, WARNING, or INFO
- **Category:** Financial, Translation, Entity, Logic, or System Error
- **Source Text:** What the source document says
- **Target Text:** What the target document says
- **Error Description:** What the difference is
- **Correction Needed:** What should be done about it
- **Confidence:** How confident the AI is in this finding (HIGH, MEDIUM, LOW)

---

## 🤖 Supported AI Models

### Cloud APIs (Recommended for Production)

#### Claude (Anthropic)
- **Claude Sonnet 4.5** ⭐ (Best Balance - Recommended)
- **Claude Opus 4.1** (Most Capable - Expensive)
- **Claude 3.5 Sonnet** (Good Balance)
- **Claude 3 Haiku** (Cheapest)

#### Gemini (Google)
- **Gemini 2.5 Pro** ⭐ (Most Capable - Recommended)
- **Gemini 2.5 Flash** (Balanced Speed/Cost)
- **Gemini 2.5 Flash-Lite** (Cheapest & Fastest)

### Local Models (Ollama - Privacy & Offline)

#### Supported Models
- **llama3** (Recommended for local use)
- **mistral** (Alternative)
- **gemma2** (Alternative)

#### Local Setup
1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull a model: `ollama pull llama3`
3. Select "Local/Ollama" in the sidebar
4. **No API key required!**

---

## 💰 API Costs (Cloud Only)

| Operation | Estimated Cost |
|-----------|----------------|
| Upload Document | ~$0.01-0.10 |
| Semantic Audit | ~$0.05-0.50 |
| Document Compare | **FREE** |
| RAG Chatbot | Pay-per-query |

**Cost-Saving Tips:**
- Use **Gemini 2.5 Flash-Lite** for lowest cost
- Use **Claude 3 Haiku** for budget Claude usage
- Use **Ollama** for completely free local processing

---

## 🛠️ Technical Architecture

```
app.py
  ├── upload_and_process_rag()  [document_upload_rag.py]
  │   ├── Extract text from PDFs (PyPDF2/MarkItDown/Docling)
  │   ├── Generate summaries (using LLM)
  │   ├── Create vector embeddings (sentence-transformers)
  │   └── Store in FAISS (local vector database)
  │
  ├── run_semantic_audit()  [document_comparison_rag.py]
  │   ├── Load documents
  │   ├── Split by [Page X] markers
  │   ├── Extract key entities (using LLM)
  │   ├── Compare sections semantically (using LLM)
  │   ├── Return RiskItems with findings
  │   └── display_audit_results() [Streamlit UI]
  │
  └── [Phase 4] agent_brain.py (Agentic Workflow)
      ├── LangGraph State Machine
      ├── Intelligent document parsing
      ├── Multi-step reasoning
      └── Self-correcting agents
```

**Central Brain:** `model_config.py`
- Single factory for LLM instantiation
- Supports: Gemini, Claude, Ollama
- Automatic fallback and error handling

---

## 🧠 Phase 4: Agentic Workflow (NEW)

### What is the Agentic Brain?

The new `agent_brain.py` module implements a **LangGraph-powered intelligent workflow** that orchestrates multiple specialized AI agents to perform complex document analysis tasks.

### Architecture

**State Machine Flow:**
```
INTAKE → PARSE → ANALYZE → AUDIT → REPORT → COMPLETE
   ↓       ↓        ↓         ↓        ↓
 ERROR → ERROR → ERROR → ERROR → ERROR
```

### Specialized Agents

1. **Document Intake Agent**
   - Validates file paths and formats
   - Initializes processing state
   - Handles error detection early

2. **Document Parser Agent**
   - Uses **MarkItDown** for Office documents (Excel, Word)
   - Uses **Docling** for complex PDFs
   - Falls back to PyPDF2 for simple PDFs

3. **Semantic Analysis Agent**
   - Generates document summaries
   - Extracts key entities (companies, dates, amounts)
   - Creates semantic embeddings

4. **Audit Agent**
   - Performs semantic comparison
   - Detects discrepancies (Error Detection mode)
   - Identifies material changes (Variance Analysis mode)

5. **Report Generator Agent**
   - Generates structured JSON reports
   - Calculates confidence scores
   - Creates human-readable summaries

### Usage Example

```python
from agent_brain import create_agentic_workflow

# Create workflow
workflow = create_agentic_workflow(
    use_local=False,
    model_name="gemini-2.5-flash",
    api_key="your_api_key"
)

# Run audit
results = workflow.run(
    doc1_path="2023_report.pdf",
    doc2_path="2024_report.pdf",
    audit_mode="variance_analysis"
)

print(results["risk_items"])
```

### Key Benefits

- **Self-Correcting:** Agents can detect failures and retry with improved strategies
- **Multi-Step Reasoning:** Complex tasks are broken into manageable steps
- **State Management:** Full transparency into the processing pipeline
- **Extensible:** Easy to add new agents or modify workflows

---

## 🔮 Future Roadmap

### Phase 5: Enterprise Integration (Microsoft Copilot Plugin)

**Goal:** Run this tool directly inside Microsoft Word/Teams as a Plugin/Add-in.

**Architecture:**
- Wrap existing Python logic into a secure REST API
- Use **Microsoft Teams Toolkit** to connect to Microsoft 365 Copilot
- **Benefit:** Audit documents without leaving the secure Microsoft Office environment

**Technical Components:**
- FastAPI backend wrapping `document_comparison_rag.py`
- Microsoft Teams Manifest for Copilot integration
- Secure authentication with Azure AD

---

### Phase 6: Advanced Vector Visualization (Apple-Style)

**Goal:** Visualize the "distance" between documents on an interactive 2D/3D map.

**Inspiration:** [Apple Embedding Atlas](https://apple.github.io/embedding-atlas/)

**Features:**
- 2D visualization of document clusters
- Interactive exploration of semantic differences
- Visual representation of document evolution over time
- Color-coded risk levels (CRITICAL = Red, WARNING = Yellow, INFO = Blue)

**Use Case:**
- Compare 5 years of annual reports on a single visual map
- See how company narrative evolved over time
- Identify semantic drift in translations

---

## 🔧 Troubleshooting

### Common Issues

**"Missing API Key"**
- For Gemini/Claude: Add your key to `.env` or enter in sidebar
- For Ollama: No key needed; make sure Ollama service is running (`ollama serve`)

**"No documents found"**
- Upload two documents in Tab 1 first
- Wait for the upload to complete (check for status messages)

**"LLM call failed"**
- **Cloud:** Verify API key is valid and you have quota remaining
- **Ollama:** Ensure Ollama service is running (`ollama serve`)
- Try a different model

**"JSON parse error"**
- The AI model's response was invalid; try again or use a different model
- For Ollama: Try a different model (mistral, gemma2)

**"Out of memory"**
- Reduce chunk size in `document_comparison_rag.py`
- Use a smaller model (Claude Haiku, Gemini Flash-Lite)
- Process smaller documents

**"Slow processing"**
- Use faster models (Gemini Flash-Lite, Claude Haiku)
- Reduce document size
- Use local Ollama for faster processing (no API latency)

---

## 🗂️ Vector Database

The app uses **FAISS** (Local Vector Store). When you process a document, it creates local folders:

- `./comparison_doc1/` - Source Vector Index
- `./comparison_doc2/` - Target Vector Index
- `./comparison_doc1_text.txt` - Extracted text (for debugging)
- `./comparison_doc2_text.txt` - Extracted text (for debugging)

**Note:** These are temporary local files. They are **NOT** sent to the cloud. You can delete them safely; the app will just ask you to process the PDF again.

---

## 🔐 Security & Privacy

### Data Privacy

- **Local Processing:** All document processing happens locally
- **Vector Storage:** FAISS stores embeddings locally (not in cloud)
- **API Calls:** Only text chunks are sent to LLM APIs (not full documents)
- **No Data Retention:** Cloud APIs don't retain your data (per Anthropic/Google policies)

### Recommended for Sensitive Documents

Use **Ollama** (Local Mode) for:
- Confidential financial reports
- Legal contracts
- Personal health information
- Any sensitive proprietary data

**Benefit:** 100% offline processing with zero data leaving your machine.

---

## 📚 Additional Documentation

- [OLLAMA_SETUP_GUIDE.md](docs/OLLAMA_SETUP_GUIDE.md) - Complete Ollama setup guide
- [COMPARISON.md](docs/COMPARISON.md) - Technical comparison details
- [Quickstart.md](docs/Quickstart.md) - Quick reference guide
- [summary.md](docs/summary.md) - System architecture summary

---

## 🧪 Testing & Validation

### Quick Test Checklist

```bash
# 1. Verify installation
pip list | grep -E "streamlit|langchain|faiss"

# 2. Check Python files compile
python -m py_compile app.py document_upload_rag.py document_comparison_rag.py agent_brain.py

# 3. Run the app
streamlit run app.py

# 4. Test each feature
# - Tab 1: Upload two sample PDFs
# - Tab 2: Run quick vector comparison
# - Tab 3: Run semantic audit (Error Detection mode)
# - Tab 3: Run semantic audit (Variance Analysis mode)
# - Tab 3: Export results as JSON
```

### Sample Test Documents

Test documents are available in `test_report/formex_sample/`:
- Use these to verify the system works correctly
- Compare translation errors and financial variances

---

## 🤝 Contributing

### Adding a New AI Provider

Edit `model_config.py`:
```python
def get_llm(use_local=True, local_model="llama3", api_key=None, cloud_model_name="gemini-2.5-flash"):
    # Add your provider here
    if cloud_model_name.startswith("gpt"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=cloud_model_name, api_key=api_key)
```

### Adjusting Audit Sensitivity

Edit `document_comparison_rag.py` prompts (lines ~316-385):
- Increase prompt detail for stricter matching
- Reduce for more lenient matching

### Changing Chunk Size

Edit `run_semantic_audit()` in `document_comparison_rag.py`:
```python
RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
```

---

## 📊 System Requirements

### Minimum Requirements
- **CPU:** 2+ cores
- **RAM:** 8GB
- **Storage:** 2GB (for dependencies + vector indices)
- **Python:** 3.10+

### Recommended Requirements (for Local Ollama)
- **CPU:** 4+ cores
- **RAM:** 16GB (32GB for larger models)
- **Storage:** 10GB+ (for Ollama models)
- **GPU:** Optional (NVIDIA GPU for faster inference)

---

## 📝 Dependencies

### Core Application
- `streamlit>=1.28.0` - Web interface
- `python-dotenv>=1.0.0` - Environment variables

### AI Framework
- `langchain>=0.1.0` - LLM orchestration
- `langchain-core>=0.3.72` - Core abstractions
- `langchain-text-splitters>=0.3.9` - Text chunking
- `langchain-community>=0.0.1` - Community integrations

### Model Providers
- `langchain-anthropic>=0.1.0` - Claude API
- `langchain-google-genai>=0.0.1` - Gemini API
- `langchain-google-vertexai>=0.0.1` - Vertex AI (optional)

### Vector Database & Search
- `faiss-cpu>=1.7.4` - Vector similarity search
- `sentence-transformers>=2.2.0` - Embeddings
- `langchain-huggingface>=0.0.1` - HuggingFace integration
- `rank_bm25>=0.2.2` - BM25 search

### Document Processing
- `PyPDF2>=3.0.0` - PDF parsing (fallback)
- `markitdown>=0.0.1a1` - Office documents (Excel, Word)
- `docling>=1.0.0` - Advanced PDF parsing

### Data Analysis
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `plotly` - Interactive charts

### Phase 4: Agentic AI
- `langgraph>=0.0.10` - Agent workflow orchestration
- `langchain-groq>=0.0.1` - Groq API (optional)

---

## 📜 Version History

| Version | Date | Phase | Description |
|---------|------|-------|-------------|
| 1.0 | Nov 29, 2025 | Phase 1 | Initial release with Cloud APIs (Claude/Gemini) |
| 1.5 | Dec 2025 | Phase 2 | Ollama integration for local processing |
| 2.0 | Dec 27, 2025 | Phase 3 | MarkItDown + Docling hybrid extraction |
| **2.5** | **Dec 27, 2025** | **Phase 4** | **Agentic workflow with LangGraph (CURRENT)** |

---

## 🎯 Current Status

**Phase:** Phase 4 - Agentic Workflow Initialization
**Status:** ✅ Ready for Development
**Date:** December 27, 2025

### Completed
- ✅ Core RAG system functional
- ✅ Hybrid document extraction (MarkItDown + Docling)
- ✅ Dual AI support (Cloud + Local)
- ✅ Two audit modes (Error Detection + Variance Analysis)
- ✅ Agent brain skeleton (`agent_brain.py`)
- ✅ LangGraph integration
- ✅ Documentation consolidation

### In Progress
- 🔄 Implementing agent logic
- 🔄 Integrating agents with existing RAG infrastructure
- 🔄 Testing agentic workflow

### Next Steps
1. Complete agent implementations in `agent_brain.py`
2. Integrate agentic workflow with Streamlit UI
3. Add visualization for agent state transitions
4. Prepare for Phase 5 (Microsoft Copilot Plugin)

---

## 📧 Contact & Support

For questions about the tool's logic or deployment, refer to:
- Technical implementation notes in code comments
- [model_config.py](model_config.py#L1) - LLM configuration
- [document_comparison_rag.py](document_comparison_rag.py#L1) - Core audit logic
- [agent_brain.py](agent_brain.py#L1) - Agentic workflow

---

**🔬 Built with Claude Code & Gemini AI**
**📄 License:** MIT (assumed - add your license)
**🌟 Star this project if you find it useful!**
