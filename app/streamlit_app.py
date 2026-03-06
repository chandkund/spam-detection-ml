import streamlit as st
import pandas as pd
import pickle

import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.sidebar.title("Spam Detection System")
st.sidebar.write("Machine Learning Project")

# Page setup
st.set_page_config(
    page_title="Spam Detection Dashboard",
    layout="wide"
)

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin1")
df = df[['v1','v2']]
df.columns = ['label','text']

st.title("📩 SMS Spam Detection Dashboard")

# ---------------------
# Dataset Info
# ---------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Messages", len(df))
col2.metric("Spam Messages", len(df[df['label']=="spam"]))
col3.metric("Ham Messages", len(df[df['label']=="ham"]))

# ---------------------
# Dataset Preview
# ---------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------
# Label Counts
# ---------------------

st.subheader("Spam vs Ham Count")

spam_count = len(df[df['label']=="spam"])
ham_count = len(df[df['label']=="ham"])

st.write("Spam Messages:", spam_count)
st.write("Ham Messages:", ham_count)

st.subheader("Word Cloud Analysis")

col1, col2 = st.columns(2)

# Spam wordcloud
with col1:

    spam_words = " ".join(df[df['label']=="spam"]['text'])

    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color="white"
    ).generate(spam_words)

    fig, ax = plt.subplots(figsize=(4,2))
    ax.imshow(wordcloud)
    ax.axis("off")

    st.pyplot(fig)

# Ham wordcloud
with col2:

    ham_words = " ".join(df[df['label']=="ham"]['text'])

    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color="white"
    ).generate(ham_words)

    fig, ax = plt.subplots(figsize=(4,2))
    ax.imshow(wordcloud)
    ax.axis("off")

    st.pyplot(fig)
# ---------------------
# Prediction Section
# ---------------------

st.subheader("Test Spam Detection")

model = pickle.load(open("models/model.pkl","rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))

message = st.text_area("Enter SMS Message")

if st.button("Predict"):

    vector = vectorizer.transform([message])
    result = model.predict(vector)[0]

    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")