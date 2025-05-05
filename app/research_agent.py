# research_agent.py
"""
Sequential product‑research agent built with **LangChain / LangGraph**.

This revision adds **robust error handling and retries** around LLM calls so a
transient network hic‑cup (e.g. `openai.APIConnectionError`) does not crash the
whole pipeline.  It also surfaces clearer log messages when a model is skipped.

Public API
==========
    run_pipeline(query: str,
                 features: list[str] | None = None,
                 models: list[str] | None = None,
                 *,
                 retries: int = 3,
                 backoff_base: float = 2.0) -> str

The two new keyword args let Streamlit adjust retry policy if desired.
"""
from __future__ import annotations

# --- Load environment variables -------------------------------------------
import time
from pathlib import Path
from dotenv import load_dotenv
import os
import argparse
import json
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import openai  # for catching APIConnectionError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
try:
    from langchain_google_genai import ChatGoogleGenAI  # optional
except ImportError:  # pragma: no cover
    ChatGoogleGenAI = None  # type: ignore

from rapidfuzz import fuzz, process
from tabulate import tabulate

# Automatically load .env one directory above this file (repo root)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Product:
    name: str
    features: Dict[str, str] = field(default_factory=dict)

    def merge(self, other: "Product") -> None:
        for k, v in other.features.items():
            if k not in self.features or not self.features[k]:
                self.features[k] = v

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def choose_llm(model_id: str):
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
    unique: List[Product] = []
    for prod in products:
        matches = process.extract(
            prod.name, [p.name for p in unique],
            scorer=fuzz.token_sort_ratio, limit=1,
        )
        if matches and matches[0][1] >= 90:
            unique[matches[0][2]].merge(prod)
        else:
            unique.append(prod)
    return unique


def _merge_products(existing: List[Product], new_batch: Sequence[dict]) -> List[Product]:
    products = existing + [Product(name=p["name"].strip(), features=p.get("features", {}))
                           for p in new_batch]
    return _fuzzy_dedupe(products)


def _products_to_markdown(products: List[Product], feature_order: List[str]) -> str:
    rows = [[prod.name] + [prod.features.get(f, "") for f in feature_order]
            for prod in products]
    return tabulate(rows, ["Product"] + feature_order, tablefmt="github")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(query: str,
                 features: List[str] | None = None,
                 models: List[str] | None = None,
                 *,
                 retries: int = 3,
                 backoff_base: float = 2.0) -> str:
    """Run the multi‑LLM enrichment loop and return a Markdown table.

    Parameters
    ----------
    query : str
        Product category / research question.
    features : list[str] | None
        Features/columns to compare. Defaults to ["price","license","platform"].
    models : list[str] | None
        Ordered list of model IDs. Defaults to ["gpt-4o"].
    retries : int, default 3
        How many times to retry an LLM call on connection errors.
    backoff_base : float, default 2.0
        Exponential back‑off base in seconds (1st retry waits base**0, then base**1, …).
    """
    feature_list = [f.strip() for f in (features or []) if f.strip()] or [
        "price", "license", "platform"]
    model_ids = models or ["gpt-4o"]

    matrix_md = ""
    collected: List[Product] = []

    for model_id in model_ids:
        chain = build_research_chain(choose_llm(model_id))
        attempt = 0
        while attempt < retries:
            try:
                response_json = chain.invoke({
                    "query": query,
                    "features": ", ".join(feature_list),
                    "current_matrix": matrix_md,
                })
                break  # Success → leave retry loop
            except openai.APIConnectionError as err:
                attempt += 1
                if attempt >= retries:
                    print(f"[ERROR] {model_id}: connection failed after {retries} attempts; skipping.")
                    response_json = None
                else:
                    wait = backoff_base ** (attempt - 1)
                    print(f"[WARN] {model_id}: connection error ({err}). Retrying in {wait}s…")
                    time.sleep(wait)
        if response_json is None:
            continue  # skip this model
        try:
            product_batch = json.loads(response_json).get("products", [])
        except json.JSONDecodeError:
            print(f"[WARN] {model_id} returned non‑JSON; skipping…")
            continue
        collected = _merge_products(collected, product_batch)
        matrix_md = _products_to_markdown(collected, feature_list)
        print(f"[INFO] After {model_id}: {len(collected)} products collected.")

    return matrix_md

# ---------------------------------------------------------------------------
# CLI for local testing
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Sequential multi‑LLM product researcher")
    parser.add_argument("--query", required=True)
    parser.add_argument("--features", default="price,license,platform")
    parser.add_argument("--models", nargs="+", default=["gpt-4o"])
    parser.add_argument("--outfile", default="matrix.md")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    md = run_pipeline(
        args.query,
        features=[f.strip() for f in args.features.split(',')],
        models=args.models,
        retries=args.retries,
    )
    Path(args.outfile).write_text(md, encoding="utf-8")
    print(f"Matrix written to {args.outfile}\n")


if __name__ == "__main__":
    _cli()
