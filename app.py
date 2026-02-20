from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Create Flask App
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------
# Simple Dataset Loader
# -----------------------------
class SimpleChatbot:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.model = None
        self.question_embeddings = None
        self.fallback_responses = [
            "I'm here to help with digital marketing. Could you please rephrase that?",
            "Let me connect you with our team. What's your business goal?",
            "I understand you're asking something. For specific queries, please contact us at info@cognitionventure.com",
            "That's a good question! Our team would love to discuss this. Call us at +91 9332015302"
        ]
        
        # Load dataset immediately
        self.load_dataset()
        
    def load_dataset(self):
        """Load dataset with multiple path attempts"""
        # Try different possible paths on Render
        possible_paths = [
            "dataset.txt",
            "/opt/render/project/src/dataset.txt",
            os.path.join(os.path.dirname(__file__), "dataset.txt")
        ]
        
        for path in possible_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and "|||" in line and not line.startswith('#'):
                            try:
                                q, a = line.split("|||", 1)
                                self.questions.append(q.strip().lower())
                                self.answers.append(a.strip())
                            except:
                                continue
                
                if self.questions:
                    logger.info(f"✅ Loaded {len(self.questions)} Q&A pairs from {path}")
                    return
            except FileNotFoundError:
                continue
        
        # If no dataset found, create default
        logger.warning("No dataset found, using defaults")
        self.create_default_dataset()
    
    def create_default_dataset(self):
        """Essential default responses"""
        defaults = [
            "hello|||Hello! Welcome to Cognition Venture. How can we help your business grow?",
            "hi|||Hi there! Cognition Venture digital marketing assistant here.",
            "who are you|||I'm the AI assistant of Cognition Venture, a digital marketing agency in Malda.",
            "services|||We offer SEO, Social Media Marketing, Web Development, and Branding.",
            "contact|||Call us: +91 9332015302 or email: info@cognitionventure.com",
            "location|||We're located in English Bazar, Malda, West Bengal, India.",
            "price|||Our packages start from ₹4,269/month for social media marketing.",
            "website|||Website development starts at ₹6,999/year.",
            "seo|||Yes, we provide professional SEO services to improve your Google rankings.",
            "ads|||Yes, we run Facebook and Google Ads campaigns.",
            "thank you|||You're welcome! Let us know if you need any help.",
            "bye|||Thank you for visiting Cognition Venture. Have a great day!"
        ]
        
        for item in defaults:
            q, a = item.split("|||", 1)
            self.questions.append(q)
            self.answers.append(a)
    
    def load_model(self):
        """Lazy-load model only when needed"""
        if self.model is None:
            try:
                logger.info("Loading AI model (first request may be slow)...")
                start = time.time()
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                
                # Create embeddings
                if self.questions:
                    self.question_embeddings = self.model.encode(
                        self.questions,
                        convert_to_tensor=True
                    )
                
                logger.info(f"✅ Model loaded in {time.time()-start:.2f} seconds")
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                return False
        return True
    
    def get_response(self, user_message):
        """Get response using semantic search or fallback"""
        if not user_message:
            return "Please type a message."
        
        # Clean message
        user_message = user_message.lower().strip()
        
        # Try semantic matching if model loads
        if self.load_model():
            try:
                # Encode user message
                user_embedding = self.model.encode(user_message, convert_to_tensor=True)
                
                # Find best match
                scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
                best_score = torch.max(scores).item()
                best_idx = torch.argmax(scores).item()
                
                # Return if confidence is good
                if best_score > 0.5:
                    return self.answers[best_idx]
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
        
        # Simple keyword matching fallback
        for q, a in zip(self.questions, self.answers):
            if any(word in user_message for word in q.split() if len(word) > 3):
                return a
        
        # Ultimate fallback
        return np.random.choice(self.fallback_responses)

# Initialize chatbot
chatbot = SimpleChatbot()

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "Cognition Venture AI is running",
        "mode": "free-tier-optimized",
        "knowledge_base": len(chatbot.questions)
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"reply": "Please type a message."})
        
        # Get response
        reply = chatbot.get_response(user_message)
        
        return jsonify({"reply": reply})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"reply": "I'm experiencing technical issues. Please contact us directly at +91 9332015302"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

# For Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
else:
    # For gunicorn
    app = app
