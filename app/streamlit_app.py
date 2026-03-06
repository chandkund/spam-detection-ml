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
st.sidebar.write("Model: Naive Bayes")

# -------------------------
# Load Dataset
# -------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/spam.csv", encoding="latin1")
    df = df[['v1','v2']]
    df.columns = ['label','text']
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
st.dataframe(df.head())

# -------------------------
# Word Cloud Section
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
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

        # Probability Score
        st.subheader("Prediction Confidence")

        st.write(f"Spam Probability: **{spam_prob:.2f}%**")
        st.write(f"Ham Probability: **{ham_prob:.2f}%**")

        st.progress(int(spam_prob))

# -------------------------
# Footer
# -------------------------

st.markdown("---")
st.write("Built with ❤️ using Streamlit and Machine Learning")