# research_agent.py
"""
Sequential product‑research agent built with **LangChain / LangGraph**.

🔧 2025‑05‑06 patch
------------------
Fixes *INVALID_PROMPT_INPUT* arising from single‑brace JSON example.
All curly braces inside the sample JSON are now **escaped as `{{` / `}}`** so
LangChain no longer thinks they are template variables.
"""
from __future__ import annotations

import time
import json
import textwrap
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import openai
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from tabulate import tabulate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
try:
    from langchain_google_genai import ChatGoogleGenAI  # optional
except ImportError:
    ChatGoogleGenAI = None  # type: ignore

# Load env vars
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# ---------------------------------------------------------------------------
# Prompt template (escaped braces)
# ---------------------------------------------------------------------------

RESEARCH_TEMPLATE = textwrap.dedent(
    """
    You are a market‑research assistant. The user wants a comparison of products
    in the following category:

    === CATEGORY ===
    {query}
    ================

    Compare across these feature columns (add sensible defaults if blank):
    {features}

    Below is the current comparison matrix in Markdown. Add **only products that
    are not already present**. If the matrix is empty, start a new list.

    --- CURRENT MATRIX ---
    {current_matrix}
    -----------------------

    ⚠️ OUTPUT FORMAT — VERY IMPORTANT ⚠️
    Respond with **valid JSON object only** (no markdown, no code fences) of the form:
      {{
        "products": [
          {{
            "name": "<Product>",
            "features": {{
              "price": "$…",
              "license": "…",
              "platform": "…"
            }}
          }}
        ]
      }}
    """
)

OUTPUT_PARSER = StrOutputParser()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Product:
    name: str
    features: Dict[str, str] = field(default_factory=dict)

    def merge(self, other: "Product") -> None:
        for k, v in other.features.items():
            if not self.features.get(k):
                self.features[k] = v

# ---------------------------------------------------------------------------
# LLM factory (JSON mode for OpenAI)
# ---------------------------------------------------------------------------

def _openai_json(model_id: str):
    return ChatOpenAI(model=model_id, temperature=0,
                      model_kwargs={"response_format": {"type": "json_object"}})

def choose_llm(model_id: str):
    if model_id.startswith("gpt-") or model_id in {"gpt-4o", "gpt-4o-mini"}:
        return _openai_json(model_id)
    if model_id.startswith("claude") or model_id.startswith("anthropic"):
        return ChatAnthropic(model=model_id, temperature=0)
    if ChatGoogleGenAI and (model_id.startswith("gemini") or model_id.startswith("google")):
        return ChatGoogleGenAI(model=model_id, temperature=0)
    raise ValueError(f"Unsupported model id: {model_id}")

# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def _build_chain(llm):
    prompt = ChatPromptTemplate.from_template(RESEARCH_TEMPLATE)
    return RunnableSequence(prompt, llm, OUTPUT_PARSER)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe(products: List[Product]) -> List[Product]:
    uniq: List[Product] = []
    for p in products:
        m = process.extract(p.name, [u.name for u in uniq], scorer=fuzz.token_sort_ratio, limit=1)
        if m and m[0][1] >= 90:
            uniq[m[0][2]].merge(p)
        else:
            uniq.append(p)
    return uniq

def _merge(existing: List[Product], batch: Sequence[dict]) -> List[Product]:
    combined = existing + [Product(name=b["name"].strip(), features=b.get("features", {})) for b in batch]
    return _dedupe(combined)


def _to_md(products: List[Product], cols: List[str]) -> str:
    rows = [[p.name] + [p.features.get(c, "") for c in cols] for p in products]
    return tabulate(rows, ["Product"] + cols, tablefmt="github")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(query: str,
                 *,
                 features: List[str] | None = None,
                 models: List[str] | None = None,
                 retries: int = 3,
                 backoff_base: float = 2.0,
                 log_raw: bool = False) -> str:
    cols = [f.strip() for f in (features or []) if f.strip()] or ["price", "license", "platform"]
    model_ids = models or ["gpt-4o"]

    matrix_md = ""
    collected: List[Product] = []

    for mid in model_ids:
        chain = _build_chain(choose_llm(mid))
        payload = {"query": query, "features": ", ".join(cols), "current_matrix": matrix_md or "<empty>"}
        raw = None
        for attempt in range(retries):
            try:
                raw = chain.invoke(payload)
                break
            except openai.APIConnectionError as e:
                wait = backoff_base ** attempt
                print(f"[WARN] {mid}: network error → retry {attempt+1}/{retries} in {wait}s…")
                time.sleep(wait)
        if raw is None:
            print(f"[ERROR] {mid}: skipped after {retries} retries.")
            continue
        if log_raw:
            print(f"[DEBUG] {mid} raw: {raw[:300]}…")
        try:
            batch = json.loads(raw).get("products", [])
        except json.JSONDecodeError as e:
            print(f"[WARN] {mid}: invalid JSON ({e}). Skipped.")
            continue
        collected = _merge(collected, batch)
        matrix_md = _to_md(collected, cols)
        print(f"[INFO] {mid}: total products {len(collected)}")

    return matrix_md

# CLI remains unchanged
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--features", default="price,license,platform")
    parser.add_argument("--log-raw", action="store_true")
    args = parser.parse_args()
    md = run_pipeline(args.query, features=[x.strip() for x in args.features.split(',')], log_raw=args.log_raw)
    Path("matrix.md").write_text(md)
    print("Matrix written to matrix.md")
