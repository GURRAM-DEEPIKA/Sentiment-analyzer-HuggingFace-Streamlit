import streamlit as st
from sentiment_utils import load_sentiment_model, get_sentiment_scores, display_sentiment_results

# ---------------------------------
# App Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Hugging Face + Streamlit Sentiment Analyzer")
st.write("This app analyzes input text and tells you whether it is positive or negative.")

# ---------------------------------
# Load Model
# ---------------------------------
MODEL_NAME = "siebert/sentiment-roberta-large-english"

with st.spinner("Loading model..."):
    classifier = load_sentiment_model(MODEL_NAME)

# ---------------------------------
# User Input
# ---------------------------------
text_input = st.text_area(
    "Enter text to analyze:",
    placeholder="Type something positive or negative..."
)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            results = get_sentiment_scores(text_input, classifier)
        display_sentiment_results(results)

# ---------------------------------
# Model Information Section
# ---------------------------------
with st.expander("ℹ️ Model Information"):
    st.write(f"**Model:** {MODEL_NAME}")
    st.write("This model is fine-tuned on 15+ datasets and achieves state-of-the-art performance.")
    st.write("It classifies text as either POSITIVE or NEGATIVE sentiment.")
