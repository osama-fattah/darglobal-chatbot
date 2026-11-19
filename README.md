# Dar Global & Wasalt Assistant

A chatbot that answers questions from public pages on **darglobal.co.uk** and **wasalt.sa**,
with a citation for every answer. It runs a free model through **OpenRouter**, is packaged with
**Docker**, and is deployed on Render.

Built by Osama Abdul Fattah for the Senior AI Engineer – Agentic Systems assessment.

## How it works

```
 darglobal.co.uk ─┐                       ┌─ BM25 keyword search ─┐
                  ├─ scraper ─ pages.jsonl ─ chunker ─┤                         ├─ RRF fusion ─ top passages
 wasalt.sa ───────┘   (robots.txt-aware)   └─ dense embeddings ┘                │
                                                                                      ▼
 user question ─ router (site + follow-up handling) ─────────────────────▶ OpenRouter free LLM
                                                                                      │
                                                     answer with [1][2] citations + source links
```

1. **Scraping** (`app/scrape.py`). A polite crawler starts from seed pages on each site, follows
   same-site links, respects `robots.txt`, waits between requests and caps the number of pages.
   Both sites are built with Next.js, so on top of the visible page text it also reads the page's
   `__NEXT_DATA__` payload, which is where listing details such as price, area and bedrooms live.
   It keeps English pages only and saves the result to `data/pages.jsonl`.
2. **Indexing** (`app/retrieval.py`). Pages are split into roughly 900-character passages with
   overlap, and each passage keeps its page title. Boilerplate repeated across pages is dropped.
   Search is hybrid: **BM25** handles exact terms (project names, cities, "338 SQM"), and a small
   **embedding model** (`bge-small-en-v1.5`, ONNX on CPU) handles meaning. Set `EMBED_MODEL` to
   `paraphrase-multilingual-MiniLM-L12-v2` for better Arabic search on a larger instance. The two result lists are merged with **Reciprocal Rank Fusion**,
   and no single page can take more than 2 of the result slots.
3. **Routing** (`app/agent.py`). Rule-based routing picks which site to search (Dar Global,
   Wasalt or both) and folds the previous question into short follow-ups like "and the price?".
   If a site filter returns too little, the search widens to both sites. Free models have tight
   daily request limits, so this step uses no LLM call and each question costs exactly one call.
4. **Answering** (`app/llm.py`). On startup the app reads OpenRouter's live model list and keeps
   only the free models, in preference order. If a model is rate-limited or errors, it moves to
   the next one. The prompt makes the model answer only from the retrieved passages, cite them,
   never invent prices or dates, and reply in the user's language.
5. **Serving** (`app/main.py`). A FastAPI backend and a single-page chat UI. It has a per-IP rate
   limit so a public link can't exhaust the API key, a `/api/status` health endpoint, and an
   optional token-protected `/api/refresh` that re-scrapes the sites.

### Data freshness
The Docker build runs the scraper once, so the deployed image starts with data. On startup the
app re-scrapes in the background if the data is older than `REFRESH_IF_OLDER_THAN_HOURS`
(default 24) and keeps serving the old index until the new one is ready.

## Run locally

```bash
cp .env.example .env            # add your OPENROUTER_API_KEY
docker build -t dg-assistant .
docker run -p 7860:7860 --env-file .env dg-assistant
# open http://localhost:7860
```

Without Docker:

```bash
pip install -r requirements.txt
python -m app.scrape --max-pages 40
OPENROUTER_API_KEY=... uvicorn app.main:app --port 7860
```

## Deploy (Render, free tier)

1. Push this repo to GitHub.
2. In Render, choose **New → Blueprint** (or **Web Service → Docker**) and select the repo.
   `render.yaml` sets up a free Docker web service with a `/health` check.
3. Add the environment variable `OPENROUTER_API_KEY` when Render asks for it.
4. Render builds the image (including the first scrape) and gives you a public URL.

Free Render services sleep after about 15 minutes without traffic, so the first request after
a pause can take up to a minute.

## Configuration

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Required. Never committed; set it as a secret. |
| `OPENROUTER_MODELS` | Optional comma-separated model list, used instead of auto-discovery. |
| `ADMIN_TOKEN` | Enables `POST /api/refresh` with header `X-Admin-Token`. |
| `REFRESH_IF_OLDER_THAN_HOURS` | Re-scrape threshold on startup (default 24). |
| `RATE_LIMIT_PER_10MIN` | Messages per IP per 10 minutes (default 25). |
| `EMBED_MODEL` | Embedding model (default `BAAI/bge-small-en-v1.5`). |
| `DISABLE_EMBEDDINGS=1` | Use BM25 only, which gives the smallest memory footprint. |

## Possible next steps
- Turn listings into structured records (price, area, city, bedrooms) and add filter tools, so
  the agent can answer questions like "villas under 1M SAR in Riyadh" by querying the data
  instead of searching text
- Add an LLM-based query planner with tool calls once a paid or higher-limit model is available
- Run the scraper on a schedule and re-index only the pages that changed
- Measure retrieval quality against a small set of test questions (hit rate, groundedness)

## Notes
- The bot only uses publicly available pages, keeps its crawl volume low and respects `robots.txt`.
- This is an independent demo for a technical assessment, not an official Dar Global or Wasalt service.
