"""
The chat pipeline: route -> retrieve -> answer with citations.

Free OpenRouter models have tight daily request limits, so routing is done
with fast rules and every user question costs exactly one LLM call.
"""
from __future__ import annotations

import re

from .llm import OpenRouter
from .retrieval import HybridIndex

SYSTEM = """You are the virtual assistant for Dar Global (a London-listed luxury real estate developer \
with projects in the UAE, Saudi Arabia, Oman, Qatar, Spain and the UK) and Wasalt (a Saudi real estate \
platform for verified property listings, projects and auctions).

Rules:
- Answer ONLY from the numbered context passages. If they don't contain the answer, say you don't have \
that information from the websites and suggest what the user could ask or where to look.
- Cite sources inline as [1], [2] matching the passage numbers. Cite every factual sentence.
- Never invent prices, dates, sizes, availability or contact details.
- Be concise and friendly. Use short bullet lists for multiple items (projects, listings).
- Prices on Wasalt are in SAR unless stated otherwise.
- Reply in the user's language (English or Arabic)."""

DAR_WORDS = re.compile(
    r"\b(dar ?global|darglobal|aida|trump|simaisma|astera|fendi|missoni|lamborghini|pagani|aston martin|"
    r"elie saab|mouawad|w hotels?|marriott|paramount|tokeni[sz]|investor|shares?|lse|london|spain|marbella|"
    r"oman|muscat|qatar|doha|dgcp|membership|golf)\b",
    re.I,
)
WASALT_WORDS = re.compile(
    r"\b(wasalt|وصلت|listing|listings|rent|rental|for sale|auction|broker|rega|fal|daily rental|"
    r"villa|apartment|land|riyadh|jeddah|dammam|khobar|makkah|mecca|madinah|taif|abha|sqm|bedroom)s?\b",
    re.I,
)


def route(question: str, history: list[dict]) -> tuple[str, str | None]:
    """Return (search_query, site filter or None)."""
    q = question.strip()
    # short follow-ups ("what about the price?") borrow the previous user turn for context
    prev_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    if prev_user and len(q.split()) <= 7:
        q = f"{prev_user} {q}"

    dar, was = bool(DAR_WORDS.search(q)), bool(WASALT_WORDS.search(q))
    site = "Dar Global" if dar and not was else "Wasalt" if was and not dar else None
    return q, site


def answer(question: str, history: list[dict], index: HybridIndex, llm: OpenRouter) -> dict:
    query, site = route(question, history)

    hits = index.search(query, k=6, site=site)
    if site and len(hits) < 3:  # filter too narrow -> widen to both sites
        hits = index.search(query, k=6)
        site = None

    if not hits:
        return {
            "answer": "I don't have any website data loaded yet, so I can't answer that. "
            "The data refresh may still be running; please try again in a minute.",
            "sources": [],
            "model": None,
            "route": {"query": query, "site": site},
        }

    context = "\n\n".join(
        f"[{i}] ({c.site}) {c.title}\nURL: {c.url}\n{c.text[:1400]}" for i, (c, _) in enumerate(hits, 1)
    )
    messages = [{"role": "system", "content": SYSTEM}]
    for m in history[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"][:1500]})
    messages.append({"role": "user", "content": f"Context passages:\n{context}\n\nQuestion: {question}"})

    text = llm.chat(messages)

    sources = []
    for i, (c, _) in enumerate(hits, 1):
        sources.append({"n": i, "site": c.site, "title": c.title or c.url, "url": c.url})
    return {
        "answer": text,
        "sources": sources,
        "model": llm.last_model,
        "route": {"query": query, "site": site or "both"},
    }
