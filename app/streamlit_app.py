# streamlit_app.py
"""
Streamlit UI for Product Research Bot
====================================

• First user message → runs research_agent.run_pipeline()
• Subsequent messages → Q‑and‑A over the saved matrix (uses GPT‑4o).
• Built‑in retries around LLM calls to avoid transient openai.APIConnectionError.

Run with:
    streamlit run app/streamlit_app.py

Environment variables are loaded via research_agent, which already imports
python‑dotenv.
"""
import time
import streamlit as st
from research_agent import run_pipeline

import openai
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Config & session initialisation
# ---------------------------------------------------------------------------

st.set_page_config(page_title="🔍 Product Research Bot", layout="wide")
st.title("🔍 Product Research Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[tuple[str, str]]
if "matrix_md" not in st.session_state:
    st.session_state.matrix_md = None

# ---------------------------------------------------------------------------
# Chat history renderer
# ---------------------------------------------------------------------------
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# ---------------------------------------------------------------------------
# Input widgets (category & optional feature list)
# ---------------------------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    prompt = st.chat_input("Describe the product category (or ask a follow‑up)…")
with c2:
    feature_str = st.text_input("Features (comma‑sep)", value="price,license,platform")

# ---------------------------------------------------------------------------
# Helper: retry wrapper for single LLM call
# ---------------------------------------------------------------------------

def ask_gpt(message: str, *, retries: int = 3, backoff_base: float = 2.0) -> str:
    attempt = 0
    while True:
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            return llm.invoke(message)
        except openai.APIConnectionError as err:
            attempt += 1
            if attempt > retries:
                raise
            wait = backoff_base ** (attempt - 1)
            st.warning(f"Connection error talking to GPT‑4o ({err}). Retrying in {wait}s…")
            time.sleep(wait)

# ---------------------------------------------------------------------------
# Main interaction logic
# ---------------------------------------------------------------------------
if prompt:
    # 1. echo user message
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. decide: first turn (run research) or follow‑up (Q‑and‑A)
    if st.session_state.matrix_md is None:
        with st.spinner("Researching across models…"):
            try:
                matrix_md = run_pipeline(
                    prompt,
                    features=[f.strip() for f in feature_str.split(',') if f.strip()],
                    models=["gpt-4o"],  # default; adjust later via UI
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.session_state.messages.append(("assistant", f"⚠️ Error: {e}"))
            else:
                if not matrix_md.strip() or "|" not in matrix_md:
                    st.warning("The models returned no products. Try a broader category or check API connectivity.")
                st.session_state.matrix_md = matrix_md
                answer = f"Here’s the initial comparison:\n\n{matrix_md}"
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    st.download_button("⬇️ Download Markdown", data=matrix_md,
                                       file_name="product_matrix.md", mime="text/markdown")
                st.session_state.messages.append(("assistant", answer))
    else:
        # Follow‑up question
        try:
            answer = ask_gpt(f"{prompt}\n\n### Existing matrix:\n{st.session_state.matrix_md}")
        except openai.APIConnectionError as err:
            answer = f"⚠️ Giving up after retries due to network error: {err}"
        st.session_state.messages.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)
