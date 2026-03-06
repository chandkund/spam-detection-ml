import pickle
from src.preprocess import clean_text

model = pickle.load(open("models/model.pkl","rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))

def predict_spam(message):

    message = clean_text(message)

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "Spam"
    else:
        return "Not Spam"