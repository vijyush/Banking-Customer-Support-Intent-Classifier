from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banking_intent.predict import predict_intent  # noqa: E402


def display_name(label: str) -> str:
    return label.replace("_", " ").title()


st.set_page_config(page_title="Banking Intent Router", page_icon="🏦", layout="centered")
st.title("Banking Support Intent Router")
st.caption(
    "Classify a customer message into one of 77 service intents and send uncertain cases for human review."
)

model_label = st.selectbox(
    "Model",
    ["MiniLM sentence embeddings", "TF-IDF baseline"],
    help="MiniLM captures semantic similarity; TF-IDF is fast and fully local after training.",
)
model_name = "transformer" if model_label.startswith("MiniLM") else "baseline"

examples = {
    "Custom message": "",
    "Card delivery": "My new card still has not arrived. Can I track it?",
    "Cash withdrawal": "The cash machine charged me but did not give me any money.",
    "Transfer status": "Why is the transfer to my friend still pending?",
}
example = st.selectbox("Try an example", list(examples))
default_text = examples[example]
message = st.text_area("Customer message", value=default_text, height=110)

if st.button("Classify intent", type="primary", use_container_width=True):
    try:
        result = predict_intent(message, model_name=model_name, top_k=3)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
    else:
        left, right = st.columns(2)
        left.metric("Predicted intent", display_name(result["predicted_intent"]))
        left.metric("Confidence signal", f"{result['confidence']:.3f}")
        right.metric("Review threshold", f"{result['handoff_threshold']:.3f}")
        right.metric("Routing", "Human review" if result["needs_human_review"] else "Automate")

        if result["needs_human_review"]:
            st.warning("The model is uncertain. Route this message to a support specialist.")
        else:
            st.success("The confidence policy accepts this prediction for automated routing.")

        st.subheader("Top intent suggestions")
        for rank, suggestion in enumerate(result["top_intents"], start=1):
            st.write(
                f"{rank}. **{display_name(suggestion['intent'])}** — score {suggestion['score']:.3f}"
            )

st.divider()
st.caption(
    "Portfolio demonstration only. Do not use this model to make financial decisions or expose private customer data."
)

