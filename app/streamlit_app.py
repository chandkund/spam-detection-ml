import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Spam Detection Dashboard",
    layout="wide"
)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("📩 Spam Detection System")
st.sidebar.write("Machine Learning Project")
st.sidebar.write("Model: Multinomial Naive Bayes")

st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit")

# -------------------------
# Load Dataset
# -------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/spam.csv", encoding="latin1")
    df = df[['v1','v2']]
    df.columns = ['label','text']

    # Fix Streamlit LargeUtf8 error
    df = df.astype(str)

    return df

df = load_data()

# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():
    model = pickle.load(open("models/model.pkl","rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------
# Title
# -------------------------

st.title("📊 SMS Spam Detection Dashboard")

# -------------------------
# Model Accuracy
# -------------------------

st.subheader("Model Performance")
st.success("Model Accuracy: **97%**")

# -------------------------
# Dataset Info
# -------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Messages", len(df))
col2.metric("Spam Messages", len(df[df['label']=="spam"]))
col3.metric("Ham Messages", len(df[df['label']=="ham"]))

# -------------------------
# Dataset Preview
# -------------------------

st.subheader("Dataset Preview")

preview_df = df.head(5).copy()
preview_df = preview_df.astype(str)

st.write(preview_df)

# -------------------------
# Word Clouds
# -------------------------

st.subheader("Word Cloud Analysis")

col1, col2 = st.columns(2)

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

# -------------------------
# Prediction Section
# -------------------------

st.subheader("🔍 Test Spam Detection")

message = st.text_area("Enter SMS Message")

# Real-time analysis
if message:

    st.subheader("Message Analysis")

    col1, col2 = st.columns(2)

    col1.metric("Message Length", len(message))
    col2.metric("Word Count", len(message.split()))

# Prediction
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")

    else:

        vector = vectorizer.transform([message])

        result = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        spam_prob = probability[1] * 100
        ham_prob = probability[0] * 100

        if result == 1:
            st.error("🚨 Spam Message Detected")
        else:
            st.success("✅ Message is Safe")

        # -------------------------
        # Spam Probability Meter
        # -------------------------

        st.subheader("Spam Probability Meter")

        st.progress(int(spam_prob))

        col1, col2 = st.columns(2)

        col1.write(f"Spam Probability: **{spam_prob:.2f}%**")
        col2.write(f"Ham Probability: **{ham_prob:.2f}%**")

# -------------------------
# Footer
# -------------------------

st.markdown("---")
st.write("Built with ❤️ using **Streamlit + Machine Learning**")