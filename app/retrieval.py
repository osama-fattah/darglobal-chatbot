"""
Hybrid retrieval over the scraped pages.

- Chunks each page into ~900-character passages with overlap
- BM25 keyword search (rank_bm25)
- Dense search with a small embedding model (fastembed, ONNX, CPU)
- Reciprocal Rank Fusion to merge the two lists
- Optional site filter ("Dar Global" / "Wasalt") chosen by the agent's router

If the embedding model can't load, it falls back to BM25 only.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

log = logging.getLogger("retrieval")

# Small English model by default so the app fits in a 512 MB free instance.
# For Arabic questions set EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_CHARS = 900
OVERLAP = 150

_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
_STOP = set(
    "a an the of to in on for and or is are was were be by with at from as it this that what which who "
    "how do does did can i you we they me my our your their about tell show list any some".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    id: int
    url: str
    site: str
    title: str
    text: str


def chunk_page(page: dict) -> list[str]:
    head = " | ".join(x for x in (page.get("title"), page.get("description")) if x)
    body = page.get("text", "")
    lines = [l for l in body.split("\n") if l.strip()]
    chunks, cur = [], ""
    for line in lines:
        if len(cur) + len(line) + 1 > CHUNK_CHARS and cur:
            chunks.append(cur)
            cur = cur[-OVERLAP:] + "\n" + line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    # prefix every chunk with the page title so each passage keeps its context
    return [f"{head}\n{c}" if head else c for c in chunks] or ([head] if head else [])


class HybridIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.vectors: np.ndarray | None = None
        self.embedder = None
        self.pages = 0

    # ---------- build ----------
    def build(self, pages_file: Path, use_embeddings: bool = True) -> None:
        pages = []
        if pages_file.exists():
            with pages_file.open(encoding="utf-8") as f:
                pages = [json.loads(l) for l in f if l.strip()]
        self.pages = len(pages)

        chunks: list[Chunk] = []
        seen: set[str] = set()
        for p in pages:
            for c in chunk_page(p):
                key = c.strip()[:400]
                if key in seen:  # drop boilerplate repeated across pages
                    continue
                seen.add(key)
                chunks.append(Chunk(len(chunks), p["url"], p["site"], p.get("title", ""), c))
        self.chunks = chunks
        if not chunks:
            self.bm25, self.vectors = None, None
            log.warning("Index is empty")
            return

        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

        self.vectors = None
        if use_embeddings:
            try:
                from fastembed import TextEmbedding

                if self.embedder is None:
                    self.embedder = TextEmbedding(model_name=EMBED_MODEL)
                vecs = np.array(list(self.embedder.embed([c.text for c in chunks], batch_size=64)), dtype=np.float32)
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
                self.vectors = vecs
            except Exception as e:  # model download blocked, etc.
                log.warning("Embeddings unavailable, using BM25 only: %s", e)
                self.embedder = None
        log.info("Indexed %d pages -> %d chunks (dense=%s)", len(pages), len(chunks), self.vectors is not None)

    # ---------- search ----------
    def search(self, query: str, k: int = 6, site: str | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks or self.bm25 is None:
            return []
        allowed = np.array([site is None or c.site == site for c in self.chunks])
        pool = 40
        rankings: list[np.ndarray] = []

        bm = self.bm25.get_scores(tokenize(query))
        bm = np.where(allowed, bm, -np.inf)
        bm_order = np.argsort(-bm)[:pool]
        rankings.append(bm_order[np.isfinite(bm[bm_order]) & (bm[bm_order] > 0)])

        if self.vectors is not None and self.embedder is not None:
            q = np.array(next(iter(self.embedder.query_embed([query]))), dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-9
            dense = self.vectors @ q
            dense = np.where(allowed, dense, -np.inf)
            rankings.append(np.argsort(-dense)[:pool])

        # Reciprocal Rank Fusion
        fused: dict[int, float] = {}
        for order in rankings:
            for rank, idx in enumerate(order):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank)
        best = sorted(fused.items(), key=lambda x: -x[1])

        # keep at most 2 passages per URL for variety
        out, per_url = [], {}
        for idx, score in best:
            c = self.chunks[idx]
            if per_url.get(c.url, 0) >= 2:
                continue
            per_url[c.url] = per_url.get(c.url, 0) + 1
            out.append((c, score))
            if len(out) >= k:
                break
        return out

    def stats(self) -> dict:
        by_site: dict[str, int] = {}
        for c in self.chunks:
            by_site[c.site] = by_site.get(c.site, 0) + 1
        return {
            "pages": self.pages,
            "chunks": len(self.chunks),
            "chunks_by_site": by_site,
            "dense_search": self.vectors is not None,
        }
