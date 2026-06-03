# Support Ticket Response RAG System

## Objective
To build a RAG for support ticket response system that retrieves information from multiple sources (internal policy document + historical tickets data) and provide to LLM to output fact-based response suggestions

## Approach
1. Data loading (PDF + tickets data CSV)
2. Text chunking
3. Create embeddings
4. Store in vector DB
5. LLM integration

### Data Preparation
- PDF: Tech Products Policy & Guidelines document (multi-page)
- CSV: 200k synthetic support tickets with issue descriptions, categories, and resolutions
- Train/test split: 80/20 stratified by category
- Combined text format: `issue_description | resolution_notes` for semantic relevance

### Embeddings + Storage
- Local embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: Chroma
- Batch processing: 5k chunks per batch to handle 161k+ documents
- Documents metadata: source (policy/tickets), ticket_id, category

### Retrieval
- Query text against vector DB using semantic similarity
- Filter by source to retrieve from policy doc or historical tickets
- Returns top-2 most relevant chunks with metadata
- Cross-source retrieval enables context from both policy and past cases

### Response Generation
- LLM provider: Groq API (llama-3.3-70b-versatile)
- Prompt structure: policy context + past cases + new ticket + instructions
- Output: fact-based response suggestions citing policy and precedent


## Usage

### Initialize Pipeline (First Run)
```bash
python main.py --mode pipeline
```
Loads data, creates chunks, and builds vector DB with embeddings (~30-60 seconds).

### Interactive Chat
```bash
python main.py --mode chat
```
Query support tickets and get LLM response suggestions. Type `exit` to quit.

## Project Structure
```
├── main.py              # CLI entry point
├── rag.py               # Core RAG functions
├── data/                # Input data (PDF + CSV)
├── chroma/              # Vector DB (auto-created)
└── README.md
```



