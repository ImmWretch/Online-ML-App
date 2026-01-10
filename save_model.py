import pickle
import os
from notmain import load_dataset, prepare_data, get_tfidf_vectorizer, train_all_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = load_dataset()
X_train, X_test, y_train, y_test = prepare_data(df)

vectorizer = get_tfidf_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

models = train_all_models(X_train_vec, y_train)
model = models["SVM"]

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved to:", BASE_DIR)