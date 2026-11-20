"""Offline tests for scraping, retrieval, routing and the model fallback.

No network: pages are small synthetic fixtures, the index runs BM25-only, and
OpenRouter calls go through an httpx MockTransport.
"""
import json

import httpx
import pytest

from app import agent, llm
from app.retrieval import HybridIndex, chunk_page, tokenize
from app.scrape import extract

PAGES = [
    {"url": "https://dg.example/aida", "site": "Dar Global", "title": "AIDA Oman",
     "description": "Golf community in Muscat", "text": "AIDA is a hilltop golf community in Muscat, Oman.\n"
     "Villas start from 1,200 SQM plots with sea views."},
    {"url": "https://dg.example/investors", "site": "Dar Global", "title": "Investor relations",
     "description": "", "text": "Dar Global shares are listed on the London Stock Exchange (LSE)."},
    {"url": "https://ws.example/villa-riyadh", "site": "Wasalt", "title": "Villa for sale in Riyadh",
     "description": "", "text": "Villa for sale in Al Malqa, Riyadh.\nPrice: 2,500,000 SAR\nArea: 338 SQM\n"
     "Bedrooms: 5"},
    {"url": "https://ws.example/apartment-jeddah", "site": "Wasalt", "title": "Apartment for rent in Jeddah",
     "description": "", "text": "Apartment for rent in Al Rawdah, Jeddah.\nPrice: 60,000 SAR per year"},
    {"url": "https://ws.example/land-riyadh", "site": "Wasalt", "title": "Land for sale in Riyadh",
     "description": "", "text": "Residential land in Al Narjis, Riyadh.\nPrice: 1,800,000 SAR\nArea: 600 SQM"},
]


@pytest.fixture
def index(tmp_path) -> HybridIndex:
    f = tmp_path / "pages.jsonl"
    f.write_text("\n".join(json.dumps(p) for p in PAGES), encoding="utf-8")
    idx = HybridIndex()
    idx.build(f, use_embeddings=False)
    return idx


class FakeLLM:
    last_model = "fake/model:free"

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def chat(self, messages: list[dict]) -> str:
        self.messages = messages
        return "Answer [1]."


# ---- scraping -------------------------------------------------------------------------------
def test_extract_reads_visible_text_and_next_data_listing_fields():
    next_data = {"props": {"pageProps": {"listing": {
        "description": "Corner villa close to schools", "price": 2500000, "bedrooms": 5,
        "image": "https://cdn.example/x.jpg", "translations": {"buy": "Buy now please"}}}}}
    html = (
        "<html><head><title>Villa in Riyadh</title><meta name='description' content='A villa'></head>"
        "<body><nav>Menu Home About</nav><main><h1>Villa for sale</h1><p>Al Malqa district</p>"
        "<a href='/other'>other</a></main>"
        f"<script id='__NEXT_DATA__' type='application/json'>{json.dumps(next_data)}</script></body></html>"
    )
    page, links = extract(html, "https://ws.example/v")
    assert page["title"] == "Villa in Riyadh" and page["description"] == "A villa"
    assert "Al Malqa district" in page["text"] and "Menu Home" not in page["text"]
    assert "description: Corner villa close to schools" in page["text"]
    assert "price: 2500000" in page["text"] and "bedrooms: 5" in page["text"]
    assert "Buy now please" not in page["text"]  # translation dictionaries are skipped
    assert "/other" in links


# ---- retrieval ------------------------------------------------------------------------------
def test_tokenize_drops_stopwords_and_keeps_arabic():
    assert tokenize("What is the price of the villa in الرياض?") == ["price", "villa", "الرياض"]


def test_chunk_page_splits_long_pages_and_prefixes_title():
    page = {"title": "T", "description": "D", "text": "\n".join(f"line {i} " + "x" * 80 for i in range(40))}
    chunks = chunk_page(page)
    assert len(chunks) > 1
    assert all(c.startswith("T | D\n") for c in chunks)
    assert all(len(c) <= 900 + 200 for c in chunks)


def test_bm25_search_finds_exact_terms_and_respects_site_filter(index):
    top, _ = index.search("338 SQM villa Riyadh", k=3)[0]
    assert top.url == "https://ws.example/villa-riyadh"
    assert all(c.site == "Dar Global" for c, _ in index.search("villa", k=5, site="Dar Global"))
    assert index.stats()["dense_search"] is False and index.stats()["pages"] == 5


def test_search_keeps_at_most_two_passages_per_page(tmp_path):
    long = {"url": "https://dg.example/long", "site": "Dar Global", "title": "Long",
            "text": "\n".join(f"golf course hole {i} " + "y" * 120 for i in range(60))}
    f = tmp_path / "pages.jsonl"
    f.write_text(json.dumps(long), encoding="utf-8")
    idx = HybridIndex()
    idx.build(f, use_embeddings=False)
    assert len(idx.chunks) > 2
    assert len(idx.search("golf course hole", k=6)) == 2


def test_empty_index_returns_no_hits(tmp_path):
    idx = HybridIndex()
    idx.build(tmp_path / "missing.jsonl", use_embeddings=False)
    assert idx.search("anything") == []


# ---- routing and answering ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "site"),
    [
        ("Where is AIDA?", "Dar Global"),
        ("Is Dar Global listed on the LSE?", "Dar Global"),
        ("Show me apartments for rent in Jeddah", "Wasalt"),
        ("What's new?", None),
    ],
)
def test_route_picks_site(question, site):
    assert agent.route(question, [])[1] == site


def test_short_follow_up_borrows_previous_question():
    history = [{"role": "user", "content": "Villas for sale in Riyadh"}, {"role": "assistant", "content": "..."}]
    query, site = agent.route("and the price?", history)
    assert query == "Villas for sale in Riyadh and the price?" and site == "Wasalt"


def test_answer_builds_numbered_context_and_sources(index):
    fake = FakeLLM()
    out = agent.answer("Price of the 338 SQM villa in Riyadh?", [], index, fake)
    assert out["answer"] == "Answer [1]." and out["model"] == "fake/model:free"
    assert out["route"]["site"] == "Wasalt"
    assert out["sources"][0] == {"n": 1, "site": "Wasalt", "title": "Villa for sale in Riyadh",
                                 "url": "https://ws.example/villa-riyadh"}
    assert fake.messages[0]["role"] == "system"
    assert "[1] (Wasalt) Villa for sale in Riyadh" in fake.messages[-1]["content"]


def test_answer_widens_to_both_sites_when_filter_is_too_narrow(index):
    out = agent.answer("Any villa listings?", [], index, FakeLLM())  # routed to Wasalt; one Wasalt hit
    assert out["route"]["site"] == "both"


def test_answer_without_data_does_not_call_the_model(tmp_path):
    idx = HybridIndex()
    idx.build(tmp_path / "missing.jsonl", use_embeddings=False)
    fake = FakeLLM()
    out = agent.answer("Hello?", [], idx, fake)
    assert out["sources"] == [] and out["model"] is None and fake.messages == []


# ---- OpenRouter client ----------------------------------------------------------------------
def _client(monkeypatch, handler) -> llm.OpenRouter:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODELS", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    client = llm.OpenRouter()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_discover_free_models_ranks_preferred_then_families(monkeypatch):
    models = [
        {"id": "google/gemma-3-27b-it:free", "pricing": {"prompt": "0"}, "context_length": 8000},
        {"id": "meta-llama/llama-3.3-70b-instruct:free", "pricing": {"prompt": "0"}, "context_length": 8000},
        {"id": "qwen/qwen-new-200b:free", "pricing": {"prompt": "0"}, "context_length": 128000},
        {"id": "qwen/qwen2.5-vl-72b:free", "pricing": {"prompt": "0"}, "context_length": 32000},
        {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025"}, "context_length": 128000},
    ]
    client = _client(monkeypatch, lambda req: httpx.Response(200, json={"data": models}))
    client.discover_free_models()
    assert client.models[:3] == [
        "meta-llama/llama-3.3-70b-instruct:free", "google/gemma-3-27b-it:free", "qwen/qwen-new-200b:free"]
    assert "openai/gpt-4o" not in client.models
    assert client.models.index("qwen/qwen2.5-vl-72b:free") > 2  # vision models go last


def test_chat_falls_back_to_next_model_on_rate_limit(monkeypatch):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.content)["model"]
        calls.append(model)
        if model == "a:free":
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    client = _client(monkeypatch, handler)
    client.models = ["a:free", "b:free"]
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert calls == ["a:free", "a:free", "b:free"] and client.last_model == "b:free"


def test_chat_skips_truncated_answers_and_raises_when_all_fail(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]})

    client = _client(monkeypatch, handler)
    client.models = ["a:free", "b:free"]
    assert client.chat([{"role": "user", "content": "hi"}]) == "cut"  # last model's partial answer is kept
    assert client.last_model == "b:free"

    client.models = ["a:free"]
    client._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(RuntimeError, match="All free models failed"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm.OpenRouter().chat([])
