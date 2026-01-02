from transformers import pipeline
import streamlit as st
import pandas as pd

# ---------------------------------------------
# Load & Cache Model
# ---------------------------------------------
@st.cache_resource
def load_sentiment_model(model_name: str):
    """
    Load and cache the sentiment analysis model from Hugging Face.
    """
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        return_all_scores=True
    )


# ---------------------------------------------
# Analyze Input Text
# ---------------------------------------------

def get_sentiment_scores(text: str, classifier):
    """
    Run sentiment analysis and return results.
    """
    if not text.strip():
        return None

    # classifier returns a list of label/score dicts
    results = classifier(text)
    return results[0]


# ---------------------------------------------
# Display/Format Output
# ---------------------------------------------
def display_sentiment_results(results):
    """
    Given the model output (list of sentiment scores dict),
    show the prediction and confidence scores in Streamlit.
    """
    if results is None:
        return

    # Get the highest scoring label
    best = max(results, key=lambda x: x["score"])

    st.subheader("Analysis Results")

    st.metric(
        "Predicted Sentiment",
        best["label"],
        f"{best['score']*100:.2f}% confidence"
    )

    st.write("### All Scores")
    df = pd.DataFrame(results)
    df["score"] = df["score"].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(df, use_container_width=True)
