from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

questions = []
answers = []

with open("dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "|||" in line:
            q, a = line.split("|||")
            questions.append(q.strip())
            answers.append(a.strip())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, X)
    best_match = similarity.argmax()

    reply = answers[best_match]

    return jsonify({"reply": reply})

app.run(host="0.0.0.0", port=5000)