import pandas as pd
from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/spam.csv", encoding="latin1")
df = df[['v1','v2']]
df.columns = ['label','text']

df['label'] = df['label'].map({'ham':0,'spam':1})

# Apply preprocessing
df['text'] = df['text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Model comparison
models = {
    "Naive Bayes":MultinomialNB(),
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "SVM":SVC()
}

for name,model in models.items():

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    print(name,accuracy_score(y_test,pred))