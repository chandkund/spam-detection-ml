
# 📧 Spam Detection System (Production-Ready ML App)

A production-oriented **machine learning system** that classifies messages as Spam or Ham using NLP, with a modular pipeline and a deployed **interactive web application**.

🔗 **Live Demo:** [https://chandkund-spam-detection-ml.streamlit.app/](https://chandkund-spam-detection-ml.streamlit.app/)

---

## 🚀 Key Highlights

* Built a **modular ML pipeline** (preprocessing → training → evaluation → inference)
* Achieved real-time predictions via **Streamlit deployment**
* Designed reusable architecture using **separation of concerns (`src/`)**
* Persisted model artifacts (`model.pkl`, `vectorizer.pkl`) for efficient inference
* Containerized application using **Docker**

---

## 🧠 Problem Statement

Spam detection is a classic NLP classification problem with real-world applications in:

* Email filtering
* Messaging platforms
* Fraud detection systems

This project focuses on building a **scalable and deployable solution**, not just a trained model.

---

## ⚙️ System Architecture

```
User Input → Streamlit UI → Preprocessing → Vectorization → Model → Prediction Output
```

### Core Components:

* **Preprocessing Layer**

  * Text cleaning, normalization, tokenization

* **Feature Engineering**

  * Vectorization using trained `vectorizer.pkl`

* **Model Layer**

  * Supervised ML classifier stored as `model.pkl`

* **Inference Layer**

  * Fast prediction pipeline via `predict.py`

* **Frontend**

  * Streamlit-based UI for real-time interaction

---

## 🗂️ Code Architecture

```bash
app/        → UI layer (Streamlit)
src/        → Core ML logic (clean, train, predict, evaluate)
models/     → Serialized artifacts
data/       → Dataset
```

This structure follows **production-grade design principles**, making the system:

* Scalable
* Maintainable
* Easily deployable

---

## 📊 Features

* Real-time spam classification
* Dataset exploration inside UI
* Word cloud visualization
* Model performance insights
* Lightweight and fast inference

---

## 🛠️ Tech Stack

* **Language:** Python
* **ML:** Scikit-learn
* **NLP:** NLTK / preprocessing techniques
* **Frontend:** Streamlit
* **Deployment:** Docker

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 🐳 Docker Deployment

```bash
docker build -t spam-detector .
docker run -p 8501:8501 spam-detector
```

---

## 📈 What Makes This Project Strong

This project demonstrates:

* Moving beyond notebooks → **production ML system**
* Understanding of **ML lifecycle (not just training)**
* Ability to **deploy and serve models**
* Clean engineering practices (modular code, reusable components)

---

## 👨‍💻 Author

**Chandan kumar (chandkund)**

---
