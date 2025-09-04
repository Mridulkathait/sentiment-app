import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import List, Dict

# ---------- one-time nltk downloads ----------
@st.cache_resource(show_spinner=False)
def _nltk_bootstrap():
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)
_nltk_bootstrap()

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))

# ---------- transformers models (cached) ----------
@st.cache_resource(show_spinner=True)
def load_sentiment_pipeline():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # labels: negative, neutral, positive
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True)
    return pipe

@st.cache_resource(show_spinner=True)
def load_emotion_pipeline():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    model_id = "joeddav/distilbert-base-uncased-go-emotions-student"  # 28 emotions + neutral
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, function_to_apply="sigmoid")
    return pipe

# ---------- helpers ----------
def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+|#\w+", " ", t)
    t = re.sub(r"[^A-Za-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = " ".join([w for w in t.split() if w not in STOPWORDS])
    return t

def run_vader(texts: List[str]) -> List[Dict]:
    sia = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        s = sia.polarity_scores(t)
        # convert to 3-class for consistency
        if s["compound"] >= 0.05:
            label = "positive"
        elif s["compound"] <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        out.append({"label": label, "score": abs(s["compound"])})
    return out

def roberta_batch(pipe, texts: List[str]) -> List[Dict]:
    # pipeline returns list of list of dicts; we pick max label
    res = pipe(texts)
    out = []
    for row in res:
        best = max(row, key=lambda x: x["score"])
        out.append({"label": best["label"].lower(), "score": float(best["score"])})
    return out

def goemo_batch(pipe, texts: List[str], threshold: float = 0.35) -> List[List[str]]:
    results = pipe(texts)
    labels_all = []
    for row in results:
        labels = [d["label"] for d in row if d["score"] >= threshold]
        # if nothing passes threshold, keep top-1
        if not labels:
            labels = [max(row, key=lambda x: x["score"])["label"]]
        labels_all.append(labels)
    return labels_all

def make_wordcloud(text_series: pd.Series, title: str):
    text = " ".join([str(t) for t in text_series.tolist()])
    wc = WordCloud(width=1100, height=500, background_color="white").generate(text)
    fig = plt.figure(figsize=(12,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    st.pyplot(fig)

# ---------- UI ----------
st.set_page_config(page_title="Social Media Sentiment & Emotion Lab", layout="wide")
st.title("🔥 Social Media Sentiment & Emotion Lab")
st.caption("Multi-model sentiment + GoEmotions, CSV analysis, dashboards, and word highlights. Built for research/demo use.")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Analysis Mode", ["Single Text", "CSV / Dataset"], index=0)
    use_roberta = st.checkbox("Use Twitter-RoBERTa Sentiment (accurate)", value=True)
    use_vader = st.checkbox("Also run VADER (fast baseline)", value=True)
    use_emotions = st.checkbox("Detect Emotions (GoEmotions)", value=True)
    emo_threshold = st.slider("Emotion threshold", 0.10, 0.80, 0.35, 0.05)

# load models only if needed
sent_pipe = load_sentiment_pipeline() if use_roberta else None
emo_pipe = load_emotion_pipeline() if use_emotions else None

# ---------- SINGLE TEXT ----------
if mode == "Single Text":
    txt = st.text_area("Paste a tweet / caption / comment:", height=120, placeholder="e.g., I absolutely love this new phone!")
    if st.button("Analyze"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            cols = st.columns(3)
            if use_roberta:
                r = roberta_batch(sent_pipe, [txt])[0]
                cols[0].metric("RoBERTa Sentiment", r["label"].title(), f"{r['score']:.2f}")
            if use_vader:
                v = run_vader([txt])[0]
                cols[1].metric("VADER Sentiment", v["label"].title(), f"{v['score']:.2f}")
            if use_emotions:
                e = goemo_batch(emo_pipe, [txt], threshold=emo_threshold)[0]
                cols[2].write("**Emotions:** " + (", ".join(e) if e else "None"))

            st.subheader("Cleaned Text")
            st.write(clean_text(txt))

# ---------- DATASET ----------
else:
    st.markdown("Upload a **CSV**. Choose the column that has the text. Optional: a date column for trend charts.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Preview:", df.head())
        text_col = st.selectbox("Text column", options=df.columns.tolist())
        date_col = st.selectbox("Date column (optional)", options=["(none)"] + df.columns.tolist(), index=0)
        run_btn = st.button("Run Analysis")

        if run_btn:
            df_proc = df.copy()
            df_proc["clean_text"] = df_proc[text_col].astype(str).apply(clean_text)

            # sentiment
            labels = []
            scores = []

            if use_roberta:
                rob = roberta_batch(sent_pipe, df_proc["clean_text"].tolist())
                labels = [r["label"] for r in rob]
                scores = [r["score"] for r in rob]
            elif use_vader:
                vad = run_vader(df_proc["clean_text"].tolist())
                labels = [v["label"] for v in vad]
                scores = [v["score"] for v in vad]
            else:
                # always have something
                vad = run_vader(df_proc["clean_text"].tolist())
                labels = [v["label"] for v in vad]
                scores = [v["score"] for v in vad]

            df_proc["sentiment"] = labels
            df_proc["confidence"] = scores

            # emotions
            if use_emotions:
                emo_labels = goemo_batch(emo_pipe, df_proc["clean_text"].tolist(), threshold=emo_threshold)
                df_proc["emotions"] = [", ".join(e) for e in emo_labels]
            else:
                df_proc["emotions"] = ""

            st.success("✅ Analysis complete.")

            # ---------- dashboards ----------
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Sentiment Distribution")
                st.plotly_chart(px.histogram(df_proc, x="sentiment", text_auto=True).update_layout(yaxis_title="Count"))

            with c2:
                st.subheader("Top Emotions")
                if use_emotions:
                    # explode multilabel
                    emo_exp = (df_proc.assign(emotion=df_proc["emotions"].str.split(", "))
                                .explode("emotion"))
                    emo_exp = emo_exp[emo_exp["emotion"].str.len() > 0]
                    st.plotly_chart(px.bar(emo_exp["emotion"].value_counts().reset_index(),
                                           x="index", y="emotion", labels={"index":"emotion","emotion":"count"}))
                else:
                    st.info("Emotion detection is off.")

            # time trend
            if date_col != "(none)":
                st.subheader("Trend Over Time")
                tmp = df_proc.copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col])
                if not tmp.empty:
                    st.plotly_chart(px.histogram(tmp, x=date_col, color="sentiment",
                                                 nbins=30, barmode="group"))
                else:
                    st.info("Could not parse dates in that column.")

            # wordclouds
            st.subheader("Word Clouds")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                make_wordcloud(df_proc["clean_text"], "All Text")
            with cc2:
                make_wordcloud(df_proc.loc[df_proc["sentiment"]=="positive","clean_text"], "Positive")
            with cc3:
                make_wordcloud(df_proc.loc[df_proc["sentiment"]=="negative","clean_text"], "Negative")

            # download results
            st.subheader("Download Results")
            out_csv = df_proc[[text_col, "clean_text", "sentiment", "confidence", "emotions"]]
            st.download_button("⬇️ Download analyzed CSV", out_csv.to_csv(index=False).encode("utf-8"),
                               file_name="analysis_results.csv", mime="text/csv")

    else:
        st.info("Upload a CSV to begin.")

