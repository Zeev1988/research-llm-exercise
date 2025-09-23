Reposetory QA Chatbot

What I built
- A small, modular system that answers questions about a local codebase using RAG:
  - Index the repo (module/class/function chunks)
  - Embed chunks using ADA2
  - Retrieve relevant snippets for a question
  - Ask GPT-4o with only those snippets and return an answer with citations

Why these design choices
- Code-aware chunking: I split Python files into module/class/function spans via AST. This is more precise than naïve fixed-size chunks and reduces irrelevant context.
- Embeddings + FAISS: Embeddings capture semantic similarity; FAISS gives fast nearest-neighbor search and scales to thousands of chunks with low latency.
- Minimal, explicit prompts: I instruct the model to only use provided snippets and to always cite file:line ranges. This keeps answers grounded and auditable.
- Local-first retrieval: I never send the whole repo. Only the top-k relevant snippets go to the model, which controls cost and improves focus.


### Key tools used
- **Azure OpenAI (embeddings + chat)**
- **FAISS (vector store)**
- **AST (code-aware chunking)**


How it works (end-to-end)
1) Indexing
   - Walk repo → parse each `.py` with AST → create chunks (with file path and line ranges)
   - Embed chunk texts with Azure ADA2 → normalize vectors → store in FAISS
   - Save aligned metadata (`metadata.jsonl`) and a small `config.json` and `manifest.json`

2) Answering a question
   - Embed the question
   - FAISS search top-k chunks → load exact code lines from disk via metadata
   - Build a constrained prompt (system + user) that includes only those snippets
   - Ask GPT-4o → return a concise answer with file:line citations

Prompts (brief rationale)
- System: “Answer using only the provided code snippets… always cite exact file:line… say if unsure.”
  - Keeps answers grounded, penalizes guessing, and enforces citations.
- User: “Question + snippets (with headers including path:start-end).”
  - Gives the model just enough context and makes citing straightforward.

Snippet format in the prompt
- Each retrieved snippet is sent with a citation header + the actual code lines:
  - Header: `# <file_path>:<start>-<end> [<symbol_type>] <symbol_name>`
  - Then the exact code lines from disk.

Usage
1) Install deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3) Build an index (outputs: faiss.index and metadata.jsonl)
```bash
python indexing_cli.py --repo /abs/path/to/repo --out /abs/path/to/res
```

4) Ask a question (reads: faiss.index and metadata.jsonl)
```bash
python ask_cli.py --index-dir /abs/path/to/res --question "What does this app do?" --k 20
```

Notes and trade-offs
- FAISS uses inner-product (cosine with normalized vectors). It’s fast and simple; for larger repos or higher precision, I could add BM25 hybrid or a reranker.
- AST-based chunking is robust for Python; for other languages I’d swap in a suitable parser.
- The system is modular by design—vector store, embedding model, and chunker can be changed independently.


