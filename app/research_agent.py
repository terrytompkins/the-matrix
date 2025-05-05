# research_agent.py
"""
Sequential product‑research agent built with **LangChain / LangGraph**, refactored
for reuse in a Streamlit chat app.

This file now exposes a single public entry‑point:

    run_pipeline(query: str,
                 features: list[str] | None = None,
                 models: list[str] | None = None) -> str

It returns a GitHub‑style Markdown table that compares products across the
requested features, after querying multiple LLMs in sequence. The code still
includes a CLI (`python research_agent.py --help`) for stand‑alone testing.

Project layout the Streamlit guide assumes:

research‑bot/
├─ app/
│  ├─ streamlit_app.py        ← Streamlit UI (see README for stub)
│  ├─ research_agent.py       ← this file
│  └─ __init__.py
├─ requirements.txt           ← add streamlit, langchain, rapidfuzz …
└─ .env.example               ← OPENAI_API_KEY=…

Requirements excerpt (put the same list in *requirements.txt*):
    streamlit>=1.35
    langchain>=0.2.4
    langchain-core>=0.2.4
    langchain-openai>=0.0.8
    langchain-anthropic>=0.0.6
    langchain-google-genai>=0.0.5  # optional – Gemini
    rapidfuzz>=3.5
    python-dotenv>=1.0
    tabulate>=0.9
    tqdm>=4.66

Environment variables:
    OPENAI_API_KEY        # for GPT‑4o or other OpenAI models
    GROK_API_KEY          # for GrokQL
    ANTHROPIC_API_KEY     # for Claude models
    GOOGLE_API_KEY        # if using Gemini
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
try:
    from langchain_google_genai import ChatGoogleGenAI  # optional
except ImportError:  # pragma: no cover
    ChatGoogleGenAI = None  # type: ignore

from rapidfuzz import fuzz, process
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Product:
    name: str
    features: Dict[str, str] = field(default_factory=dict)

    def merge(self, other: "Product") -> None:
        """Merge non-empty features from another Product with same name."""
        for k, v in other.features.items():
            if k not in self.features or not self.features[k]:
                self.features[k] = v


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def choose_llm(model_id: str):
    """Return a LangChain chat model instance for the given model id."""
    if model_id.startswith("gpt-") or model_id in {"gpt-4o", "gpt-4o-mini"}:
        return ChatOpenAI(model=model_id, temperature=0)
    if model_id.startswith("claude") or model_id.startswith("anthropic"):
        return ChatAnthropic(model=model_id, temperature=0)
    if ChatGoogleGenAI and (model_id.startswith("gemini") or model_id.startswith("google")):
        return ChatGoogleGenAI(model=model_id, temperature=0)
    raise ValueError(f"Unsupported model id: {model_id}")


RESEARCH_TEMPLATE = textwrap.dedent(
    """
    You are a market‑research assistant. The user wants a comparison of products
    in the following category:

    === CATEGORY ===
    {query}
    ================

    The comparison should cover these features (add sensible defaults if blank):
    {features}

    Here are products we have already gathered (if any) in Markdown table form.
    Only add **new** products that are missing.

    --- CURRENT MATRIX ---
    {current_matrix}
    -----------------------

    Return ONLY a JSON array called `products`, where each element has:
        name: string – product name
        features: object – keys = feature names, values = strings
    """
)

OUTPUT_PARSER = StrOutputParser()


def build_research_chain(llm):
    prompt = ChatPromptTemplate.from_template(RESEARCH_TEMPLATE)
    return RunnableSequence(prompt, llm, OUTPUT_PARSER)


# ---------------------------------------------------------------------------
# Core pipeline logic
# ---------------------------------------------------------------------------

def _fuzzy_dedupe(products: List[Product]) -> List[Product]:
    """Deduplicate on fuzzy name similarity (threshold 90)."""
    unique: List[Product] = []
    for prod in products:
        matches = process.extract(
            prod.name, [p.name for p in unique],
            scorer=fuzz.token_sort_ratio, limit=1
        )
        if matches and matches[0][1] >= 90:
            idx = matches[0][2]
            unique[idx].merge(prod)
        else:
            unique.append(prod)
    return unique


def _merge_products(existing: List[Product], new_batch: Sequence[dict]) -> List[Product]:
    products = existing + [Product(name=p["name"].strip(), features=p.get("features", {}))
                           for p in new_batch]
    return _fuzzy_dedupe(products)


def _products_to_markdown(products: List[Product], feature_order: List[str]) -> str:
    rows = []
    for prod in products:
        row = [prod.name] + [prod.features.get(f, "") for f in feature_order]
        rows.append(row)
    headers = ["Product"] + feature_order
    return tabulate(rows, headers, tablefmt="github")


# ---------------------------------------------------------------------------
# Public API for Streamlit & others
# ---------------------------------------------------------------------------

def run_pipeline(query: str,
                 features: List[str] | None = None,
                 models: List[str] | None = None) -> str:
    """Run the multi‑LLM research loop and return a Markdown table string."""
    feature_list = [f.strip() for f in (features or []) if f.strip()]
    if not feature_list:
        feature_list = ["price", "license", "platform"]
    model_ids = models or ["gpt-4o"]

    matrix_md = ""  # start empty
    collected: List[Product] = []

    for model_id in model_ids:
        llm = choose_llm(model_id)
        chain = build_research_chain(llm)
        response_json = chain.invoke({
            "query": query,
            "features": ", ".join(feature_list),
            "current_matrix": matrix_md,
        })
        try:
            product_batch = json.loads(response_json).get("products", [])
        except json.JSONDecodeError:
            print(f"[WARN] {model_id} returned non‑JSON; skipping…")
            continue
        collected = _merge_products(collected, product_batch)
        matrix_md = _products_to_markdown(collected, feature_list)
        print(f"[INFO] After {model_id}: matrix has {len(collected)} products.")

    return matrix_md


# ---------------------------------------------------------------------------
# CLI wrapper for local testing
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Sequential multi‑LLM product researcher")
    parser.add_argument("--query", required=True, help="Product category to research")
    parser.add_argument("--features", default="price,license,platform",
                        help="Comma‑separated feature list")
    parser.add_argument("--models", nargs="+", default=["gpt-4o"],
                        help="Model IDs in order (e.g. gpt-4o claude-3-opus)")
    parser.add_argument("--outfile", default="matrix.md", help="Output Markdown path")
    args = parser.parse_args()

    md = run_pipeline(args.query,
                      features=[f.strip() for f in args.features.split(',')],
                      models=args.models)

    with open(args.outfile, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"\nMatrix written to {args.outfile}\n")


if __name__ == "__main__":
    _cli()
