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

import hashlib
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
MAX_CHUNKS_PER_PAGE = 25

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
    @staticmethod
    def _cache_paths(pages_file: Path) -> tuple[Path, Path]:
        return pages_file.with_name("index_vectors.npy"), pages_file.with_name("index_meta.json")

    def build(self, pages_file: Path, use_embeddings: bool = True) -> None:
        """Build the index into local variables, then swap it in (safe while serving)."""
        pages = []
        if pages_file.exists():
            with pages_file.open(encoding="utf-8") as f:
                pages = [json.loads(l) for l in f if l.strip()]
        digest = hashlib.sha1(pages_file.read_bytes()).hexdigest() if pages_file.exists() else ""

        chunks: list[Chunk] = []
        seen: set[str] = set()
        for p in pages:
            for c in chunk_page(p)[:MAX_CHUNKS_PER_PAGE]:
                key = c.strip()[:400]
                if key in seen:  # drop boilerplate repeated across pages
                    continue
                seen.add(key)
                chunks.append(Chunk(len(chunks), p["url"], p["site"], p.get("title", ""), c))
        if not chunks:
            log.warning("Index is empty")
            self.chunks, self.bm25, self.vectors, self.pages = [], None, None, len(pages)
            return

        bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        # BM25 is ready: serve with it now, add dense vectors below
        self.vectors = None
        self.chunks, self.bm25, self.pages = chunks, bm25, len(pages)
        log.info("BM25 ready: %d pages -> %d chunks", len(pages), len(chunks))

        if not use_embeddings:
            return
        vec_path, meta_path = self._cache_paths(pages_file)
        try:
            from fastembed import TextEmbedding

            if self.embedder is None:
                self.embedder = TextEmbedding(model_name=EMBED_MODEL)
            vecs = None
            if vec_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("digest") == digest and meta.get("model") == EMBED_MODEL and meta.get("n") == len(chunks):
                    vecs = np.load(vec_path)
                    log.info("Loaded cached embeddings")
            if vecs is None:
                vecs = np.array(list(self.embedder.embed([c.text for c in chunks], batch_size=32)), dtype=np.float32)
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
                try:
                    np.save(vec_path, vecs)
                    meta_path.write_text(json.dumps({"digest": digest, "model": EMBED_MODEL, "n": len(chunks)}))
                except OSError as e:
                    log.warning("Could not cache embeddings: %s", e)
            if self.chunks is chunks:  # still the current index
                self.vectors = vecs
        except Exception as e:  # model download blocked, etc.
            log.warning("Embeddings unavailable, using BM25 only: %s", e)
            self.embedder = None
        log.info("Indexed %d chunks (dense=%s)", len(chunks), self.vectors is not None)

    # ---------- search ----------
    def search(self, query: str, k: int = 6, site: str | None = None) -> list[tuple[Chunk, float]]:
        chunks, bm25, vectors = self.chunks, self.bm25, self.vectors
        if not chunks or bm25 is None:
            return []
        if vectors is not None and len(vectors) != len(chunks):
            vectors = None
        allowed = np.array([site is None or c.site == site for c in chunks])
        pool = 40
        rankings: list[np.ndarray] = []

        bm = bm25.get_scores(tokenize(query))
        bm = np.where(allowed, bm, -np.inf)
        bm_order = np.argsort(-bm)[:pool]
        rankings.append(bm_order[np.isfinite(bm[bm_order]) & (bm[bm_order] > 0)])

        if vectors is not None and self.embedder is not None:
            q = np.array(next(iter(self.embedder.query_embed([query]))), dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-9
            dense = vectors @ q
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
            c = chunks[idx]
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


if __name__ == "__main__":
    # Pre-compute embeddings at Docker build time so the small runtime box doesn't have to.
    logging.basicConfig(level=logging.INFO)
    from .scrape import OUT_FILE
    HybridIndex().build(OUT_FILE)
