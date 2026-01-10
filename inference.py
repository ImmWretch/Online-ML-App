import pickle
import os
from notmain import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

_model = pickle.load(open("model.pkl", "rb"))
_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(text):
    text = clean_text(text)
    vec = _vectorizer.transform([text])
    pred = _model.predict(vec)[0]
    return "SPAM" if pred == 1 else "HAM"
