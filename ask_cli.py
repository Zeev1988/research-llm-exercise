import json, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from azure.identity import get_bearer_token_provider, EnvironmentCredential
from openai import AzureOpenAI
from embedding import AzureEmbeddingsClient, AzureChatClient, make_azure_clients


def load_resources(index_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index = faiss.read_index(str(index_dir / "faiss.index"))
    metadata: List[Dict[str, Any]] = []
    with (index_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    return index, metadata


def get_relevant_chunks(index: faiss.Index, metadata: List[Dict[str, Any]], query_vec: np.ndarray, k: int = 8):
    D, I = index.search(query_vec.astype(np.float32), k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        hits.append((float(score), idx, meta))
    return hits


def read_snippet(repo_root: Path, meta: Dict[str, Any]) -> str:
    file_path = Path(meta["file_path"])
    if not file_path.is_absolute():
        file_path = repo_root / file_path
    if not file_path.exists():
        return f"# Missing file: {file_path}"
    start, end = int(meta["start_line"]), int(meta["end_line"])
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start)
    end = min(len(lines), end if end >= start else start)
    snippet = "\n".join(lines[start - 1 : end])
    header = f"# {meta['file_path']}:{start}-{end} [{meta['symbol_type']}] {meta['symbol_name']}"
    return f"{header}\n{snippet}"


def build_messages(question: str, snippets: List[str]) -> List[Dict[str, str]]:
    context = "\n\n".join(snippets)
    system = (
        "You are a code assistant. Answer using only the provided code snippets. "
        "Always cite exact file:line ranges you used. If unsure, say you are unsure."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Code snippets (with citations):\n{context}\n\n"
        "Respond with a concise answer followed by bullet list of citations."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer_question(index_dir: Path, question: str, top_k: int = 20) -> str:
    index, metadata = load_resources(index_dir)

    chat_client = AzureChatClient()
    emb_client = AzureEmbeddingsClient()

    q_vec = emb_client.embed_texts([question])
    hits = get_relevant_chunks(index, metadata, q_vec, k=top_k)

    snippets = [read_snippet(Path("/"), meta) for _, _, meta in hits]
    messages = build_messages(question, snippets)

    content = chat_client.chat(messages)
    return content or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=str, required=True, help="Path to /res (index output dir)")
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    out = answer_question(Path(args.index_dir), args.question, top_k=args.k)
    print(out)


if __name__ == "__main__":
    main()
