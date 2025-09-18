from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

from embedding.azure import AzureEmbeddingsClient
from vectorstore.faiss_store import FaissWriter
from .walker import get_python_files_from_repo
from .chunker import chunk_python_file



console = Console()

EXCLUDE_DIR_NAMES = ["tests"]

def build_chunks_fo_python_repo(repo_path: Path):
    """Walk the repository and produce semantic code chunks per Python file.

    - Uses get_python_files_from_repo to list candidate files (with exclusions).
    - For each file, chunk_python_file returns module/class/function chunks.
    - Aggregates all chunks into a single list.
    """
    all_chunks = []
    files = get_python_files_from_repo(repo_path) # TODO: add option for other formats
    files = [f for f in files if not any(r'/'+ sub + r'/' in str(f) for sub in EXCLUDE_DIR_NAMES)]
    for f in files:
        try:
            chunks = chunk_python_file(f)
            all_chunks.extend(chunks)
        except Exception:
            # Skip files that fail to parse; indexing should be resilient.
            pass
    return all_chunks


def _prepare_chunk_text_for_embedding(chunk, max_chars: int = 4000) -> str:
    """Compose a compact, informative text representation of a chunk for embedding.

    Includes a header with location and type, optional docstring, and body text.
    Truncates to max_chars to keep embedding requests efficient.
    """
    header = f"[{chunk.symbol_type}] {chunk.symbol_name}:{chunk.start_line}-{chunk.end_line}"
    doc = chunk.docstring or ""
    body = chunk.text or ""
    combined = f"{header}\n\n{doc}\n\n{body}"
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def index_repository(
    repo_path: Path,
    output_dir: Path
) -> Path:
    """Build a vector index for a repository.

    Steps:
    1) Parse and chunk Python files.
    2) Create Azure OpenAI embeddings for chunks.
    3) Persist vectors (FAISS) and metadata (JSONL) to output directory.
    """
    repo_path = repo_path.resolve()
    repo_name = repo_path.name
    out_dir = output_dir or (Path.cwd() / ".repo_index" / repo_name)

    console.log(f"Indexing repo: {repo_path}")
    console.log(f"Output dir: {out_dir}")

    chunks = build_chunks_fo_python_repo(repo_path)
    if not chunks:
        raise RuntimeError("No chunks produced; check repository path and filters.")

    texts = [_prepare_chunk_text_for_embedding(c) for c in chunks]
    metas: List[Dict[str, Any]] = []
    for c in chunks:
        metas.append(
            {
                "id": c.id,
                "file_path": c.file_path,
                "symbol_name": c.symbol_name,
                "symbol_type": c.symbol_type,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
        )

    emb_client = AzureEmbeddingsClient()
    vectors = emb_client.embed_texts(texts)

    writer = FaissWriter()
    writer.add(vectors, metas)
    writer.save(out_dir)

    console.log(f"Indexed {len(chunks)} chunks → {out_dir}")
    return out_dir
