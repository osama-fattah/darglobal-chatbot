"""
Polite crawler for public pages on darglobal.co.uk and wasalt.sa.

- Respects robots.txt
- Stays on each site's own domain and on English pages
- Caps pages per site and waits between requests
- Pulls clean page text plus readable strings from Next.js __NEXT_DATA__
  (both sites are Next.js, so listing details often live there)

Output: data/pages.jsonl, one JSON object per page:
  {"url", "site", "title", "description", "text", "scraped_at"}

Run:  python -m app.scrape            (defaults)
      python -m app.scrape --max-pages 80
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlparse
from urllib import robotparser

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger("scraper")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_FILE = DATA_DIR / "pages.jsonl"

USER_AGENT = (
    "Mozilla/5.0 (compatible; DarGlobalAssignmentBot/1.0; "
    "candidate technical assessment; respects robots.txt)"
)

SITES = {
    "Dar Global": {
        "base": "https://darglobal.co.uk",
        "seeds": [
            "/", "/about", "/projects", "/investor", "/why-invest", "/press",
            "/luxury-golf-communities", "/hospitality", "/commercial",
            "/tokenization", "/exclusive-membership", "/get-in-touch",
            "/landing-page/aida", "/lp/trump-international-qatar",
            "/rayana/trump-mansions",
            "/partners/aston-martin", "/partners/elie-saab", "/partners/fendi-casa",
            "/partners/lamborghini", "/partners/marriott-residences-aida-oman",
            "/partners/missoni", "/partners/mouawad", "/partners/pagani",
            "/partners/the-trump-organization", "/partners/w-hotels",
            "/partners/paramount",
        ],
        # skip non-English duplicates, files and tracking links
        "skip": re.compile(r"/(ar|es|ru|zh|fr|de)(/|$)|\.(pdf|jpg|jpeg|png|webp|svg|mp4|zip)$|/cdn-cgi/", re.I),
        "max_pages": 120,
    },
    "Wasalt": {
        "base": "https://wasalt.sa",
        "seeds": [
            "/en", "/en/about", "/en/corporate", "/en/s/wasalt-business",
            "/en/projects", "/en/auction",
            "/en/properties-for-sale-in-saudi-arabia",
            "/en/properties-for-rent-in-saudi-arabia",
            "/en/properties-for-sale-in-riyadh",
            "/en/properties-for-sale-in-jeddah",
            "/en/properties-for-rent-in-riyadh",
            "/en/properties-for-rent-in-jeddah",
        ],
        # English pages only; skip deep pagination and files
        "skip": re.compile(r"^(?!/en(/|$))|[?&]page=([3-9]|\d{2,})|\.(pdf|jpg|jpeg|png|webp|svg|mp4|zip)$", re.I),
        "max_pages": 150,
    },
}

NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "header", "footer", "nav"]


def _clean(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _next_data_strings(soup: BeautifulSoup, limit: int = 250, max_chars: int = 6000) -> list[str]:
    """Collect human-readable strings from the page's own Next.js data (pageProps only).

    Translation dictionaries and app-wide config are skipped; they repeat on every page
    and would swamp the index.
    """
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return []
    try:
        data = json.loads(tag.string)
    except json.JSONDecodeError:
        return []

    out: list[str] = []
    seen: set[str] = set()
    total = 0
    skip_keys = {"buildId", "image", "images", "url", "href", "src", "icon", "slug", "id", "_id", "uuid",
                 "token", "seo", "meta", "schema", "breadcrumbs", "footer", "header", "menu", "navigation"}
    skip_sub = ("i18n", "translation", "locale", "messages", "dictionary", "_next", "initialstate", "config")

    def walk(node, key=""):
        nonlocal total
        if len(out) >= limit or total >= max_chars:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                kl = k.lower()
                if k in skip_keys or any(x in kl for x in skip_sub):
                    continue
                walk(v, k)
        elif isinstance(node, list):
            for v in node[:40]:
                walk(v, key)
        elif isinstance(node, str):
            s = BeautifulSoup(node, "lxml").get_text(" ") if "<" in node else node
            s = s.strip()
            if (
                4 <= len(s) <= 1500
                and " " in s
                and not s.startswith(("http", "/", "{", "data:"))
                and s not in seen
            ):
                seen.add(s)
                line = f"{key}: {s}" if key and len(key) < 30 else s
                out.append(line)
                total += len(line)
        elif isinstance(node, (int, float)) and key and not isinstance(node, bool):
            if any(w in key.lower() for w in ("price", "area", "bed", "bath", "room", "size", "floor")):
                out.append(f"{key}: {node}")

    walk(data.get("props", {}).get("pageProps", {}))
    return out


def extract(html: str, url: str) -> tuple[dict, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    description = (meta.get("content") or "").strip() if meta else ""

    links = [a.get("href") for a in soup.find_all("a", href=True)]
    next_strings = _next_data_strings(soup)

    for t in soup(NOISE_TAGS):
        t.decompose()
    main = soup.find("main") or soup.body or soup
    body_text = _clean(main.get_text("\n"))

    # add Next.js strings that are not already in the visible text
    extra = [s for s in next_strings if s.split(": ", 1)[-1] not in body_text]
    text = body_text
    if extra:
        text += "\n" + "\n".join(extra)

    return {"title": title, "description": description, "text": _clean(text)}, links


def crawl_site(client: httpx.Client, name: str, cfg: dict, max_pages: int, delay: float) -> list[dict]:
    base = cfg["base"]
    host = urlparse(base).netloc
    rp = robotparser.RobotFileParser()
    try:
        r = client.get(urljoin(base, "/robots.txt"))
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
    except httpx.HTTPError:
        rp.parse([])

    queue = deque(urljoin(base, s) for s in cfg["seeds"])
    seen: set[str] = set()
    pages: list[dict] = []

    while queue and len(pages) < max_pages:
        url = urldefrag(queue.popleft())[0].rstrip("/") or base
        if url in seen:
            continue
        seen.add(url)
        path = urlparse(url).path + (("?" + urlparse(url).query) if urlparse(url).query else "")
        if cfg["skip"].search(path or "/"):
            continue
        if not rp.can_fetch(USER_AGENT, url):
            log.info("robots.txt disallows %s", url)
            continue
        try:
            resp = client.get(url)
        except httpx.HTTPError as e:
            log.warning("fetch failed %s: %s", url, e)
            continue
        finally:
            time.sleep(delay)
        if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
            continue

        page, links = extract(resp.text, url)
        if len(page["text"]) > 200:
            pages.append({
                "url": str(resp.url),
                "site": name,
                **page,
                "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            log.info("[%s] %d/%d %s", name, len(pages), max_pages, url)

        for href in links:
            nxt = urljoin(url, href)
            p = urlparse(nxt)
            if p.scheme in ("http", "https") and p.netloc.removeprefix("www.") == host.removeprefix("www."):
                if nxt not in seen:
                    queue.append(nxt)

    return pages


def main(max_pages: int | None = None, delay: float = 0.6) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_pages: list[dict] = []
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    with httpx.Client(headers=headers, timeout=25, follow_redirects=True) as client:
        for name, cfg in SITES.items():
            pages = crawl_site(client, name, cfg, max_pages or cfg["max_pages"], delay)
            log.info("%s: %d pages", name, len(pages))
            all_pages.extend(pages)

    if not all_pages:
        log.error("No pages scraped; keeping any existing data file")
        return 1
    tmp = OUT_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for p in all_pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(OUT_FILE)
    log.info("Saved %d pages to %s", len(all_pages), OUT_FILE)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=None, help="cap per site")
    ap.add_argument("--delay", type=float, default=0.6, help="seconds between requests")
    a = ap.parse_args()
    raise SystemExit(main(a.max_pages, a.delay))
