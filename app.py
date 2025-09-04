import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Title
st.title("Social Media Sentiment Analysis")

# Text input
user_text = st.text_area("Enter a text/tweet for sentiment analysis:")

# Dummy sentiment prediction (replace with your ML model later)
if st.button("Analyze"):
    if any(word in user_text.lower() for word in ["good", "love", "happy", "thank"]):
        st.success("😊 Positive Sentiment")
    elif any(word in user_text.lower() for word in ["bad", "sad", "angry", "hate"]):
        st.error("😡 Negative Sentiment")
    else:
        st.info("😐 Neutral Sentiment")

# Wordcloud Visualization (demo words)
st.subheader("Word Cloud Example")
text = "love good day work thank happy go time night one"
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)
