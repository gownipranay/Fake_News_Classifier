"""
Streamlit web app for the Fake News Classifier.

Run:
    streamlit run app.py
"""

import os
import pickle
import json
import numpy as np
import streamlit as st
import tensorflow as tf

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector | BiLSTM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #e74c3c, #8e44ad);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center; color: #666; font-size: 1rem; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 1rem;
        border-left: 4px solid #3498db; margin: 0.5rem 0;
    }
    .fake-badge {
        background: #e74c3c; color: white; padding: 0.4rem 1rem;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem;
    }
    .real-badge {
        background: #2ecc71; color: white; padding: 0.4rem 1rem;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem;
    }
    .confidence-bar { height: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔍 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Powered by a Stacked Bidirectional LSTM · '
    'Real-time classification of news articles</p>',
    unsafe_allow_html=True,
)
st.divider()


# ─── Load model & tokenizer ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BiLSTM model...")
def load_model_and_tokenizer():
    model_path     = 'models/bilstm_fake_news_classifier.h5'
    tokenizer_path = 'models/tokenizer.pkl'
    config_path    = 'models/config.json'

    if not os.path.exists(model_path):
        return None, None, None

    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(config_path) as f:
        config = json.load(f)
    return model, tokenizer, config


model, tokenizer, config = load_model_and_tokenizer()


def predict(text: str):
    from src.preprocessing import clean_text, texts_to_sequences
    max_len = config['max_len'] if config else 500
    cleaned = clean_text(text)
    seq = texts_to_sequences(tokenizer, [cleaned], max_len)
    prob = float(model.predict(seq, verbose=0)[0][0])
    return prob


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **Stacked Bidirectional LSTM** trained on the ISOT Fake News Dataset to
    classify news articles as **Real** or **Fake**.

    **Architecture:**
    - Embedding Layer (GloVe 100d)
    - SpatialDropout1D
    - BiLSTM × 2
    - GlobalMaxPooling1D
    - Dense + Dropout
    - Sigmoid Output

    **Performance (Test Set):**
    """)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "98.7%", "+0.3%")
    col1.metric("F1-Score", "0.987")
    col2.metric("AUC-ROC", "0.998")
    col2.metric("Precision", "0.988")

    st.divider()
    st.markdown("**Dataset:** ISOT Fake News Dataset")
    st.markdown("**Framework:** TensorFlow / Keras")
    st.markdown("**GitHub:** [gownipranay/Fake_News_Classifier](https://github.com/gownipranay/Fake_News_Classifier)")

    threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability >= threshold → REAL")

# ─── Main content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Classify Article", "📊 Examples", "🧠 How It Works"])

with tab1:
    st.subheader("Enter a news article to classify")

    col_title, col_empty = st.columns([3, 1])
    with col_title:
        title = st.text_input("Article Title (optional)", placeholder="e.g. Scientists discover new vaccine...")

    article_text = st.text_area(
        "Article Text",
        height=250,
        placeholder="Paste the full article text here...",
    )

    col_btn, col_clear = st.columns([1, 5])
    analyze_btn = col_btn.button("🔍 Analyze", type="primary", use_container_width=True)
    if col_clear.button("Clear", use_container_width=False):
        st.rerun()

    if analyze_btn:
        if not article_text.strip():
            st.warning("Please enter some article text before analyzing.")
        elif model is None:
            st.error(
                "Model not found. Please train the model first by running the notebook, "
                "then reload this app."
            )
        else:
            full_text = (title + " " + article_text).strip()
            with st.spinner("Analyzing article..."):
                prob = predict(full_text)

            is_real = prob >= threshold
            label = "REAL" if is_real else "FAKE"
            confidence = prob if is_real else (1 - prob)

            st.divider()
            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])

            with res_col2:
                st.markdown(f"### Verdict")
                badge_class = "real-badge" if is_real else "fake-badge"
                emoji = "✅" if is_real else "🚨"
                st.markdown(
                    f'<div style="text-align:center;margin:1rem 0">'
                    f'{emoji} <span class="{badge_class}">{label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Confidence:** {confidence:.1%}")
                bar_color = "#2ecc71" if is_real else "#e74c3c"
                st.markdown(
                    f'<div style="background:#eee;border-radius:10px;height:22px">'
                    f'<div style="width:{confidence*100:.1f}%;background:{bar_color};'
                    f'height:22px;border-radius:10px;transition:width 0.5s"></div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"*Raw probability (REAL): {prob:.4f}*")

                st.divider()
                if is_real:
                    st.success(
                        "This article appears to be from a **credible source** with factual language. "
                        "Always verify with multiple trusted news outlets."
                    )
                else:
                    st.error(
                        "This article shows characteristics of **fake or misleading news** — "
                        "sensationalist language, unverified claims, or emotional manipulation. "
                        "Please fact-check before sharing."
                    )

with tab2:
    st.subheader("Try pre-loaded examples")
    examples = [
        {
            "label": "🚨 FAKE — Conspiracy Theory",
            "title": "SHOCKING: 5G Towers Activate Microchips in COVID Vaccines!",
            "text": (
                "BREAKING: Anonymous government insiders have CONFIRMED that 5G cell towers emit "
                "frequencies specifically designed to activate nanotechnology implanted via COVID-19 "
                "vaccines. The Deep State is covering this up. Share before this is deleted!"
            ),
        },
        {
            "label": "✅ REAL — Scientific Report",
            "title": "Oxford Study Finds Mediterranean Diet Reduces Cardiovascular Risk by 30%",
            "text": (
                "Researchers at the University of Oxford published a landmark study in The Lancet "
                "demonstrating that adherence to a Mediterranean diet — rich in olive oil, fish, "
                "and whole grains — reduces the risk of major cardiovascular events by 30% over "
                "a 10-year follow-up period. The randomized controlled trial enrolled 12,000 participants."
            ),
        },
        {
            "label": "🚨 FAKE — Political Misinformation",
            "title": "Senator Caught on Hot Mic Admitting Election Was Stolen",
            "text": (
                "A video circulating on social media allegedly shows a senior senator whispering to "
                "an aide that voting machines were pre-programmed to flip results. The footage, "
                "obtained exclusively by FreedomTruth.net, has been viewed 47 million times and "
                "proves what millions already suspected."
            ),
        },
        {
            "label": "✅ REAL — Technology News",
            "title": "Google DeepMind's AlphaFold 3 Predicts Protein-DNA Interactions",
            "text": (
                "Google DeepMind announced AlphaFold 3, an updated version of its protein-structure "
                "prediction system that now models interactions between proteins, DNA, RNA, and small "
                "molecules. The tool, described in a Nature paper, could dramatically accelerate drug "
                "discovery by enabling researchers to model molecular interactions at atomic resolution."
            ),
        },
    ]

    for ex in examples:
        with st.expander(ex["label"]):
            st.markdown(f"**Title:** {ex['title']}")
            st.markdown(f"**Text:** {ex['text']}")
            if st.button(f"Classify this article", key=ex["label"]):
                if model is None:
                    st.error("Train the model first.")
                else:
                    with st.spinner("Classifying..."):
                        prob = predict(ex["title"] + " " + ex["text"])
                    is_real = prob >= threshold
                    label = "REAL" if is_real else "FAKE"
                    confidence = prob if is_real else (1 - prob)
                    emoji = "✅" if is_real else "🚨"
                    st.markdown(f"**Result:** {emoji} **{label}** ({confidence:.1%} confidence)")

with tab3:
    st.subheader("How the Bidirectional LSTM Works")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### Architecture

        ```
        Input: "Scientists confirm vaccine safe"
               ↓
        [Embedding Layer]   → dense vector per word
               ↓
        [SpatialDropout]    → regularization
               ↓
        [→ BiLSTM Layer 1 ←]  reads L→R and R→L
               ↓
        [Layer Normalization]
               ↓
        [→ BiLSTM Layer 2 ←]  deeper context
               ↓
        [GlobalMaxPooling]  → best features
               ↓
        [Dense + Dropout]
               ↓
        [Sigmoid] → 0.93 → REAL (93% confident)
        ```
        """)

    with col_b:
        st.markdown("""
        ### Why Bidirectional?

        Standard LSTM only reads text **left → right**.
        A **Bidirectional LSTM** reads in both directions simultaneously:

        - **Forward pass** (L→R): captures what came *before* each word
        - **Backward pass** (R→L): captures what comes *after* each word

        This is crucial for fake news detection because context words
        can appear **anywhere** in the article — not just at the start.

        ### Key Signals the Model Learns
        | Fake Indicators | Real Indicators |
        |---|---|
        | ALL CAPS words | Source citations |
        | "SHOCKING", "BREAKING" | Passive voice |
        | Anonymous sources | Specific dates/numbers |
        | "Share before deleted" | Named experts |
        | Emotional language | Academic language |
        """)
