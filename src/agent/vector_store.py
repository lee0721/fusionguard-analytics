#!/usr/bin/env python3
"""Hybrid document retriever supporting FAISS vector search with TF-IDF fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

CHUNK_PATTERN = re.compile(r"\n{2,}")

# Optional heavy dependencies (graceful fallbacks handled at runtime)
try:  # pragma: no cover - import guarded for environments without faiss
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - handled dynamically
    faiss = None  # type: ignore

try:  # pragma: no cover - import guarded for environments without sentence-transformers
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - handled dynamically
    SentenceTransformer = None  # type: ignore


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    """Container for retrieval results."""

    text: str
    source: Path
    score: float


class DocumentStore:
    """Load markdown documentation and surface relevant snippets."""

    def __init__(
        self,
        root_dirs: Sequence[Path] | None = None,
        *,
        min_chars: int = 120,
        max_chars: int = 1200,
        use_embeddings: bool = True,
        embedding_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.root_dirs = list(root_dirs) if root_dirs else [Path("docs")]
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.embedding_model = embedding_model

        self._chunks: List[str] = []
        self._sources: List[Path] = []
        self._tfidf_matrix = None
        self._faiss_index = None
        self._encoder: Optional[SentenceTransformer] = None  # type: ignore[assignment]
        self._embeddings: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None

        embeddings_available = bool(
            use_embeddings and faiss is not None and SentenceTransformer is not None
        )
        self.backend = "faiss" if embeddings_available else "tfidf"

        if self.backend == "faiss":
            try:
                self._encoder = SentenceTransformer(self.embedding_model)  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - fallback when model cannot be loaded offline
                self.backend = "tfidf"

        if self.backend == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words="english")

        self._build_index()

    def _load_files(self) -> Iterable[Tuple[Path, str]]:
        for root in self.root_dirs:
            if not root.exists():
                continue
            for path in root.rglob("*.md"):
                try:
                    yield path, path.read_text()
                except UnicodeDecodeError:
                    continue

    def _split_into_chunks(self, text: str) -> List[str]:
        raw_chunks = [section.strip() for section in CHUNK_PATTERN.split(text) if section.strip()]
        chunks: List[str] = []
        for chunk in raw_chunks:
            if len(chunk) <= self.max_chars:
                if len(chunk) >= self.min_chars:
                    chunks.append(chunk)
                continue
            # Break long chunk into approximate sentences
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            buffer: List[str] = []
            current_len = 0
            for sentence in sentences:
                if current_len + len(sentence) > self.max_chars and buffer:
                    merged = " ".join(buffer).strip()
                    if len(merged) >= self.min_chars:
                        chunks.append(merged)
                    buffer = [sentence]
                    current_len = len(sentence)
                else:
                    buffer.append(sentence)
                    current_len += len(sentence)
            if buffer:
                merged = " ".join(buffer).strip()
                if len(merged) >= self.min_chars:
                    chunks.append(merged)
        return chunks

    def _build_index(self) -> None:
        for path, text in self._load_files():
            for chunk in self._split_into_chunks(text):
                self._chunks.append(chunk)
                self._sources.append(path)

        if not self._chunks:
            # Fallback to avoid downstream errors
            self._chunks.append("No documentation available. Respond with general project guidance.")
            self._sources.append(Path("<empty>"))

        if self.backend == "faiss":
            embeddings = self._encoder.encode(  # type: ignore[union-attr]
                self._chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings = embeddings.astype(np.float32)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # type: ignore[union-attr]
            index.add(embeddings)
            self._faiss_index = index
            self._embeddings = embeddings
        else:
            self._tfidf_matrix = self.vectorizer.fit_transform(self._chunks)  # type: ignore[union-attr]

    def query(self, question: str, top_k: int = 4) -> List[RetrievedChunk]:
        total_chunks = len(self._chunks)
        if total_chunks == 0:
            return []
        limit = min(top_k, total_chunks)
        results: List[RetrievedChunk] = []

        if self.backend == "faiss" and self._faiss_index is not None:
            query_vec = self._encoder.encode(  # type: ignore[union-attr]
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            scores, indices = self._faiss_index.search(query_vec, limit)
            for idx, score in zip(indices[0], scores[0]):
                chunk_idx = int(idx)
                results.append(
                    RetrievedChunk(
                        text=self._chunks[chunk_idx],
                        source=self._sources[chunk_idx],
                        score=float(score),
                    )
                )
            return results

        if self._tfidf_matrix is None or self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([question])
        cosine_similarities = linear_kernel(query_vec, self._tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[::-1][:limit]
        for idx in top_indices:
            score = float(cosine_similarities[idx])
            results.append(
                RetrievedChunk(
                    text=self._chunks[idx],
                    source=self._sources[idx],
                    score=score,
                )
            )
        return results

    @property
    def sources(self) -> Sequence[Path]:
        return self._sources
