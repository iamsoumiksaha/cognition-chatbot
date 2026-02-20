from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# create app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# load dataset
questions = []
answers = []

with open("dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "|||" in line:
            q, a = line.split("|||")
            questions.append(q.strip())
            answers.append(a.strip())

# train vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if user_message.strip() == "":
        return jsonify({"reply": "Please type a message."})

    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, X)
    best_match = similarity.argmax()

    reply = answers[best_match]

    return jsonify({"reply": reply})


# required for Render hosting
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

