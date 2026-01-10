from notmain import load_dataset
from notmain import prepare_data
from notmain import get_tfidf_vectorizer
from notmain import train_all_models
from notmain import predict_message
from notmain import evaluate_models
from notmain import plot_graph

def main():
    print("Loading dataset......")
    df = load_dataset()
    print("Preparing Data .......")
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Vectorizing......")
    vectorizer = get_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Model......")
    models = train_all_models(X_train_vec, y_train)

    print("Evaluting Models......")
    results = evaluate_models(models, X_test_vec, y_test)
    plot_graph(results)

    print("Enter message:")
    S = input(">")
    print("Message you have entered is : ", S)
    print()
    algorithm = "SVM"
    model = models[algorithm]
    ANS = predict_message(model, vectorizer, S)
    print("Prediction: ", ANS)


if __name__ == "__main__":
    main()