from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# create flask app
app = Flask(__name__)

# allow website requests (CORS)
CORS(app, resources={r"/*": {"origins": "*"}})

# load chatbot dataset
questions = []
answers = []

with open("dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "|||" in line:
            q, a = line.split("|||")
            questions.append(q.strip())
            answers.append(a.strip())

# train text vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)


# health check route (important for Render)
@app.route("/")
def home():
    return "Cognition Venture AI is running"


# chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_message = data.get("message", "").strip()

    if user_message == "":
        return jsonify({"reply": "Please type a message."})

    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, X)
    best_match = similarity.argmax()

    reply = answers[best_match]

    return jsonify({"reply": reply})
