import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from src.preprocess import clean_text


# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin1")

df = df[['v1','v2']]
df.columns = ['label','text']

# Convert label
df['label'] = df['label'].map({'ham':0,'spam':1})

# Clean text
df['text'] = df['text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train,y_train)

# Save model
pickle.dump(model,open("models/model.pkl","wb"))
pickle.dump(vectorizer,open("models/vectorizer.pkl","wb"))

print("Model saved successfully")