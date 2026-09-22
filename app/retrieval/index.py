"""
retrieval/index.py — Corpus loader, section-aware chunker, and index manager.

The loader is designed as an adapter: swap `CorpusLoader.load()` to return
`list[Document]` from any corpus format (PDF, JSONL, TXT, etc.) in < 10 min.

TODO(SCHEMA): Adapt load() once real corpus format is confirmed.
Each chunk gets a stable citation ID: "Doc_12 §2.3" style.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.schemas import Chunk


# ─────────────────────────────────────────────────────────────────────────────
# Raw document representation (pre-chunking)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A raw document loaded from the corpus."""
    doc_id: str          # e.g. "Doc_01", "Policy_HR_2024"
    title: str
    sections: list["Section"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Section:
    """A heading + body section within a document."""
    section_id: str      # e.g. "§2", "§2.3"
    heading: str
    body: str
    parent_section_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Corpus Loader — ADAPTER LAYER
# ─────────────────────────────────────────────────────────────────────────────

class CorpusLoader:
    """
    Loads documents from the corpus directory.

    Supported formats (auto-detected by file extension):
    - .jsonl: Each line is {"doc_id":..., "title":..., "sections":[{"id":..., "heading":..., "body":...}]}
    - .json:  Same structure as JSONL but one JSON object or array
    - .txt:   Treated as a single section document
    - .md:    Markdown with ## headings treated as section boundaries

    TODO(SCHEMA): Add PDF loader once real corpus format is confirmed.
    The adapter contract: load() -> list[Document]
    """

    def __init__(self, corpus_path: str) -> None:
        self._path = Path(corpus_path)

    def load(self) -> list[Document]:
        """Load all documents from the corpus directory."""
        if not self._path.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {self._path}. "
                "Mount the corpus volume or set CORPUS_PATH. "
                "See README for instructions."
            )

        docs: list[Document] = []
        for f in sorted(self._path.rglob("*")):
            if f.is_file():
                try:
                    loaded = self._load_file(f)
                    if loaded:
                        docs.extend(loaded)
                except Exception as e:
                    print(f"[CorpusLoader] Warning: failed to load {f}: {e}")
        return docs

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl(path)
        elif suffix == ".json":
            return self._load_json(path)
        elif suffix in (".md", ".markdown"):
            return [self._load_markdown(path)]
        elif suffix == ".txt":
            return [self._load_txt(path)]
        return []

    def _load_jsonl(self, path: Path) -> list[Document]:
        docs = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                docs.append(self._parse_raw_doc(raw, path, i))
        return docs

    def _load_json(self, path: Path) -> list[Document]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return [self._parse_raw_doc(item, path, i) for i, item in enumerate(raw)]
        return [self._parse_raw_doc(raw, path, 0)]

    def _parse_raw_doc(self, raw: dict, path: Path, idx: int) -> Document:
        doc_id = raw.get("doc_id") or raw.get("id") or f"Doc_{path.stem}_{idx:03d}"
        title = raw.get("title") or raw.get("name") or doc_id
        sections_raw = raw.get("sections") or raw.get("chunks") or []
        sections = []
        for j, s in enumerate(sections_raw):
            sections.append(Section(
                section_id=s.get("id") or s.get("section_id") or f"§{j+1}",
                heading=s.get("heading") or s.get("title") or f"Section {j+1}",
                body=s.get("body") or s.get("text") or s.get("content") or "",
            ))
        if not sections and raw.get("text"):
            sections = [Section(section_id="§1", heading="Content", body=raw["text"])]
        return Document(doc_id=doc_id, title=title, sections=sections, metadata=raw.get("metadata", {}))

    def _load_markdown(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        doc_id = f"Doc_{path.stem}"
        title = doc_id
        # Extract title from first # heading
        first_heading = re.match(r"^#\s+(.+)", text, re.MULTILINE)
        if first_heading:
            title = first_heading.group(1).strip()

        # Split on ## headings
        sections: list[Section] = []
        parts = re.split(r"^(#{1,3}\s+.+)$", text, flags=re.MULTILINE)
        current_heading = "Introduction"
        current_body_parts: list[str] = []
        sec_num = 1

        for part in parts:
            if re.match(r"^#{1,3}\s+", part):
                if current_body_parts:
                    body = "\n".join(current_body_parts).strip()
                    if body:
                        sections.append(Section(
                            section_id=f"§{sec_num}",
                            heading=current_heading,
                            body=body,
                        ))
                        sec_num += 1
                current_heading = re.sub(r"^#+\s+", "", part).strip()
                current_body_parts = []
            else:
                current_body_parts.append(part)

        if current_body_parts:
            body = "\n".join(current_body_parts).strip()
            if body:
                sections.append(Section(
                    section_id=f"§{sec_num}",
                    heading=current_heading,
                    body=body,
                ))

        return Document(doc_id=doc_id, title=title, sections=sections)

    def _load_txt(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        doc_id = f"Doc_{path.stem}"
        return Document(
            doc_id=doc_id,
            title=doc_id,
            sections=[Section(section_id="§1", heading="Content", body=text)],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section-aware chunker
# ─────────────────────────────────────────────────────────────────────────────

class SectionAwareChunker:
    """
    Converts Documents into Chunks with stable citation IDs.

    Strategy:
    1. Respect section boundaries — never split a chunk across sections.
    2. If a section body exceeds chunk_size_tokens, split it on paragraph/sentence
       boundaries and number the sub-chunks: "Doc_12 §2 [1/3]".
    3. Each chunk carries doc_id + section_id so the citation ID is stable.

    R7 spike: section-aware chunking is compared against naive fixed-size in bench/.
    """

    def __init__(self, chunk_size_tokens: int = 300, overlap_tokens: int = 40) -> None:
        self._chunk_size = chunk_size_tokens
        self._overlap = overlap_tokens

    def chunk(self, docs: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            for section in doc.sections:
                body = section.body.strip()
                if not body:
                    continue
                sub_chunks = self._split_section(body)
                total = len(sub_chunks)
                for i, text in enumerate(sub_chunks):
                    if total == 1:
                        chunk_id = f"{doc.doc_id} {section.section_id}"
                    else:
                        chunk_id = f"{doc.doc_id} {section.section_id} [{i+1}/{total}]"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        section=section.section_id,
                        text=text,
                        token_count=self._count_tokens(text),
                        metadata={
                            "heading": section.heading,
                            "title": doc.metadata.get("title", doc.doc_id),
                        },
                    ))
        return chunks

    def _split_section(self, text: str) -> list[str]:
        """Split section body into sub-chunks respecting token limits."""
        tokens = self._tokenize(text)
        if len(tokens) <= self._chunk_size:
            return [text]

        # Split on paragraphs first, then sentences
        paragraphs = re.split(r"\n{2,}", text)
        chunks: list[str] = []
        current_tokens: list[str] = []

        for para in paragraphs:
            para_tokens = self._tokenize(para)
            if len(current_tokens) + len(para_tokens) <= self._chunk_size:
                current_tokens.extend(para_tokens)
            else:
                if current_tokens:
                    chunks.append(self._detokenize(current_tokens))
                if len(para_tokens) > self._chunk_size:
                    # Split on sentences
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    sent_buf: list[str] = []
                    for sent in sentences:
                        s_tokens = self._tokenize(sent)
                        if len(sent_buf) + len(s_tokens) <= self._chunk_size:
                            sent_buf.extend(s_tokens)
                        else:
                            if sent_buf:
                                chunks.append(self._detokenize(sent_buf))
                            sent_buf = s_tokens[-self._overlap:] if self._overlap else []
                    if sent_buf:
                        chunks.append(self._detokenize(sent_buf))
                    current_tokens = []
                else:
                    current_tokens = list(para_tokens[-self._overlap:]) if self._overlap else []
                    current_tokens.extend(para_tokens)

        if current_tokens:
            chunks.append(self._detokenize(current_tokens))
        return chunks if chunks else [text]

    def _tokenize(self, text: str) -> list[str]:
        """Whitespace tokenisation — fast and LangChain-free."""
        return text.split()

    def _detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def _count_tokens(self, text: str) -> int:
        return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Index Manager — builds and serves the combined index at startup
# ─────────────────────────────────────────────────────────────────────────────

class IndexManager:
    """
    Orchestrates corpus loading, chunking, and building of BM25 + FAISS indices.
    Built once at startup; all retrieval components reference the same manager.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._chunk_map: dict[str, Chunk] = {}
        self._loaded = False
        self._build_time_s: float = 0.0

    def build(self, corpus_path: str) -> None:
        """Load corpus, chunk, and build all indices. Call once at startup."""
        t0 = time.perf_counter()
        cfg = get_settings()

        # 1. Load corpus
        loader = CorpusLoader(corpus_path)
        docs = loader.load()
        print(f"[IndexManager] Loaded {len(docs)} documents from {corpus_path}")

        # 2. Chunk
        chunker = SectionAwareChunker(
            chunk_size_tokens=cfg.chunk_size_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
        )
        self._chunks = chunker.chunk(docs)
        self._chunk_map = {c.chunk_id: c for c in self._chunks}
        print(f"[IndexManager] Created {len(self._chunks)} chunks")

        # 3. Build retrieval indices (imported lazily to avoid circular imports)
        from app.retrieval.bm25 import BM25Index
        from app.retrieval.dense import DenseIndex

        self._bm25 = BM25Index()
        self._bm25.build(self._chunks)

        self._dense = DenseIndex()
        self._dense.build(self._chunks)

        self._build_time_s = time.perf_counter() - t0
        self._loaded = True
        print(f"[IndexManager] Index built in {self._build_time_s:.2f}s")

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def chunk_map(self) -> dict[str, Chunk]:
        return self._chunk_map

    @property
    def bm25(self) -> "BM25Index":  # type: ignore[name-defined]
        self._require_loaded()
        return self._bm25

    @property
    def dense(self) -> "DenseIndex":  # type: ignore[name-defined]
        self._require_loaded()
        return self._dense

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def build_time_s(self) -> float:
        return self._build_time_s

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunk_map.get(chunk_id)

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("IndexManager.build() has not been called yet.")


# ── Module-level singleton ──────────────────────────────────────────────────
_index_manager: Optional[IndexManager] = None


def get_index_manager() -> IndexManager:
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager
