"""Simple local retrieval layer for rubric and transcript evidence.

Day 1 uses TF-IDF retrieval because it is transparent, fast, local, and easy to
debug. Later versions can swap this implementation for sentence-transformer
embeddings, Chroma, FAISS, or a LangChain retriever without changing the rest
of the pipeline.

This module retrieves two kinds of evidence:
    1. Rubric/guideline chunks from `knowledge_base/`
    2. Transcript chunks from the selected video
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import RetrievedEvidence


@dataclass
class LocalTfidfRetriever:
    """A tiny TF-IDF retriever over text chunks."""

    chunks: list[str]
    sources: list[str]

    def __post_init__(self) -> None:
        """Fit a TF-IDF vectorizer over the provided chunks."""

        if len(self.chunks) != len(self.sources):
            raise ValueError("chunks and sources must have the same length.")
        if not self.chunks:
            raise ValueError("At least one chunk is required.")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, k: int = 5) -> list[RetrievedEvidence]:
        """Return the top-k chunks for a query.

        Args:
            query: Search query.
            k: Number of chunks to return.

        Returns:
            Ranked evidence chunks.
        """

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:k]

        results: list[RetrievedEvidence] = []
        for rank, idx in enumerate(top_idx, start=1):
            results.append(
                RetrievedEvidence(
                    source=self.sources[idx],
                    text=self.chunks[idx],
                    score=float(sims[idx]),
                    rank=rank,
                )
            )
        return results


def chunk_text(text: str, source: str, max_chars: int = 700) -> tuple[list[str], list[str]]:
    """Split text into simple paragraph-sized chunks.

    Args:
        text: Text to chunk.
        source: Source label attached to each chunk.
        max_chars: Approximate maximum chunk length.

    Returns:
        Tuple of chunk texts and matching source labels.
    """

    raw_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    sources: list[str] = []

    for part in raw_parts:
        if len(part) <= max_chars:
            chunks.append(part)
            sources.append(source)
        else:
            # Fallback: split long paragraphs into sentence-like fragments.
            sentences = [s.strip() for s in part.replace(". ", ".\n").split("\n") if s.strip()]
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= max_chars:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        chunks.append(current)
                        sources.append(source)
                    current = sentence
            if current:
                chunks.append(current)
                sources.append(source)

    return chunks, sources


def build_knowledge_retriever(knowledge_dir: str | Path = "knowledge_base") -> LocalTfidfRetriever:
    """Build a retriever over markdown files in the knowledge base."""

    chunks: list[str] = []
    sources: list[str] = []

    for path in sorted(Path(knowledge_dir).glob("*.md")):
        file_chunks, file_sources = chunk_text(path.read_text(encoding="utf-8"), source=str(path))
        chunks.extend(file_chunks)
        sources.extend(file_sources)

    return LocalTfidfRetriever(chunks=chunks, sources=sources)


def build_transcript_retriever(transcript: str, source: str = "transcript") -> LocalTfidfRetriever:
    """Build a retriever over the selected video's transcript."""

    chunks, sources = chunk_text(transcript, source=source, max_chars=500)
    return LocalTfidfRetriever(chunks=chunks, sources=sources)


def retrieve_for_metrics(
    metric_names: Iterable[str],
    transcript: str,
    knowledge_dir: str | Path = "knowledge_base",
    k_per_source: int = 2,
) -> tuple[list[RetrievedEvidence], list[RetrievedEvidence]]:
    """Retrieve rubric context and transcript evidence for metric interpretation.

    Args:
        metric_names: Metric names to use as retrieval queries.
        transcript: Transcript for the selected video.
        knowledge_dir: Directory containing rubric/guideline markdown files.
        k_per_source: Number of chunks per query/source.

    Returns:
        A tuple of (rubric_context, transcript_evidence).
    """

    knowledge_retriever = build_knowledge_retriever(knowledge_dir)
    transcript_retriever = build_transcript_retriever(transcript)

    rubric_results: list[RetrievedEvidence] = []
    transcript_results: list[RetrievedEvidence] = []

    for metric_name in metric_names:
        query = metric_name.replace("_", " ")
        rubric_results.extend(knowledge_retriever.search(query, k=k_per_source))
        transcript_results.extend(transcript_retriever.search(query, k=1))

    return _dedupe_results(rubric_results), _dedupe_results(transcript_results)


def _dedupe_results(results: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    """Remove duplicate chunks while preserving rank-like ordering."""

    seen: set[tuple[str, str]] = set()
    deduped: list[RetrievedEvidence] = []

    for result in results:
        key = (result.source, result.text)
        if key not in seen:
            seen.add(key)
            deduped.append(result)

    return deduped
