import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import faiss


class FaissWriter:
    """Tiny helper to write a FAISS index and aligned metadata.

    Usage:
      writer = FaissWriter()
      writer.add(vectors, metadatas)
      writer.save(Path("/path/to/out"))
    """

    def __init__(self) -> None:
        self.index = None
        self.dim = None
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        if vectors.shape[0] != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")
        if vectors.size == 0:
            return
        if self.index is None:
            self.dim = int(vectors.shape[1])
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def save(self, out_dir: Path) -> None:
        if self.index is None or self.dim is None:
            raise RuntimeError("No data to save; index is empty.")
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_dir / "faiss.index"))

        with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        # Index artifacts: faiss.index and metadata.jsonl only (kept simple)


