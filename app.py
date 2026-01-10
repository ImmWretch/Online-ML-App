from flask import Flask, request, jsonify
from inference import predict

app = Flask(__name__)
@app.route("/")
def home():
    return "Spam Detection API is Runnning"
@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data["text"]
    result = predict(text)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
    