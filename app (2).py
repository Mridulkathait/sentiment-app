import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from utils import preprocess_text

st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")

st.title("🔥 Social Media Sentiment Analyzer")
st.write("Analyzing **1.6M tweets (Sentiment140 dataset)** directly from the internet!")

# Load dataset directly from GitHub raw
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasets/sentiment140/master/training.1600000.processed.noemoticon.csv"
    df = pd.read_csv(
        url,
        encoding="latin-1",
        names=["target", "ids", "date", "flag", "user", "text"]
    )
    return df

st.info("Loading dataset... (may take ~1–2 minutes)")
df = load_data()

st.success(f"✅ Dataset Loaded | Total Tweets: {len(df)}")

# Show sample data
st.subheader("📊 Sample Data")
st.write(df.sample(5))

# Map labels
df["target"] = df["target"].map({0: "Negative", 2: "Neutral", 4: "Positive"})

# Sentiment distribution
st.subheader("📊 Sentiment Distribution")
sentiment_counts = df["target"].value_counts()
fig, ax = plt.subplots()
sentiment_counts.plot(kind="bar", color=["red", "blue", "green"], ax=ax)
st.pyplot(fig)

# WordCloud
st.subheader("☁️ WordCloud for Tweets")
text = " ".join(df["text"].astype(str).sample(5000))  # limit for performance
wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# User Input Sentiment
st.subheader("📝 Try Your Own Sentence")
user_input = st.text_area("Enter text here:")
if user_input:
    cleaned = preprocess_text(user_input)
    analysis = TextBlob(cleaned)
    if analysis.sentiment.polarity > 0:
        st.success("Sentiment: Positive 😀")
    elif analysis.sentiment.polarity < 0:
        st.error("Sentiment: Negative 😡")
    else:
        st.info("Sentiment: Neutral 😐")
