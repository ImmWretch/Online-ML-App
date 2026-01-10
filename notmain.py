#all the libraries recquired:
import pandas as pd
import matplotlib.pyplot as plt

import re
import string
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import pickle

#loading data:
def load_dataset(path=r"C:\Users\Akshat\Desktop\Over Here\Projects\Finalized\project 4\data\spam.csv"):
    df = pd.read_csv(path)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df


#preprocessing data:
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def prepare_data(df):
    df['message'] = df['message'].apply(clean_text)
    X = df['message']
    y = df['label']
    return train_test_split(X, y,test_size=0.2, random_state=42, stratify=y)


#training the models:
def train_all_models(X_train_vec, y_train):
    
    models = {}
    
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    models["Naive Bayes"] = nb

    svm = LinearSVC()
    svm.fit(X_train_vec, y_train)
    models["SVM"] = svm

    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(X_train_vec, y_train)
    models["Logistic Regression"] = logreg

    return models

#vectorizing data:
def get_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features = 3000,
        stop_words = 'english',
        ngram_range=(1,2)
    )

#single prediction:
def predict_message(model, vectorizer, text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    return "SPAM" if pred == 1 else "HAM"


#evaluating models:
def evaluate_models(models, X_test_vec, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] ={"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return results
    
#plotting:
def plot_graph(results):
    names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, accuracies)
    plt.title("MOdel Accuracy Comparison")
    plt.ylabel("Acc")
    plt.ylim(0, 1)
    plt.show()
