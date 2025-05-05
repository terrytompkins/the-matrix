import os, streamlit as st
from app.research_agent import run_pipeline   # we’ll wrap main() into a function

# --- Page config -----------------------------------------------------------
st.set_page_config(page_title="Product Research Bot", layout="wide")
st.title("🔍 Product Research Bot")

# --- Chat history in Session State ----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display history ------------------------------------------------------
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# --- User input -----------------------------------------------------------
prompt = st.chat_input("Describe the product category (or ask a follow‑up)…")
if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Decide whether it’s a *research* or *follow‑up* question ----------
    if "matrix_md" not in st.session_state:
        # First turn → run the multi‑LLM pipeline
        with st.spinner("Researching across models…"):
            matrix_md = run_pipeline(prompt)
        st.session_state.matrix_md = matrix_md
        answer = f"Here’s the initial comparison:\n\n{matrix_md}"
        # Download button
        st.download_button("⬇️ Download Markdown",
                           data=matrix_md,
                           file_name="product_matrix.md",
                           mime="text/markdown")
    else:
        # Follow‑up Q&A: feed (question + matrix) to GPT‑4o
        from langchain_openai import ChatOpenAI
        qa_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        answer = qa_llm.invoke(f"{prompt}\n\n### Matrix:\n{st.session_state.matrix_md}")

    st.session_state.messages.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
