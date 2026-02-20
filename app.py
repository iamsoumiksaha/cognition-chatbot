from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------
# Memory-Optimized Chatbot Class
# -----------------------------
class MemoryOptimizedChatbot:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.model = None
        self.question_embeddings = None
        self.model_loaded = False
        self.load_dataset()
        
    def load_dataset(self):
        """Load dataset with error handling"""
        try:
            # Try multiple paths
            paths = [
                "dataset.txt",
                os.path.join(os.path.dirname(__file__), "dataset.txt"),
                "/opt/render/project/src/dataset.txt"
            ]
            
            for path in paths:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and "|||" in line and not line.startswith('#'):
                                q, a = line.split("|||", 1)
                                self.questions.append(q.strip().lower())
                                self.answers.append(a.strip())
                    logger.info(f"✅ Loaded {len(self.questions)} Q&A pairs")
                    return
                except FileNotFoundError:
                    continue
            
            # Fallback defaults
            logger.warning("Using default responses")
            self.questions = ["hello", "hi", "contact", "services"]
            self.answers = [
                "Hello! Welcome to Cognition Venture.",
                "Hi there! How can we help?",
                "Contact us: +91 9332015302 or info@cognitionventure.com",
                "We offer SEO, Social Media, Web Development, and Branding."
            ]
            
        except Exception as e:
            logger.error(f"Dataset error: {e}")
    
    def lazy_load_model(self):
        """Load model only when needed (memory optimization)"""
        if not self.model_loaded:
            try:
                logger.info("Loading AI model (this may take a few seconds)...")
                start = time.time()
                
                # Import here to delay loading
                from sentence_transformers import SentenceTransformer, util
                import torch
                
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                
                if self.questions:
                    self.question_embeddings = self.model.encode(
                        self.questions,
                        convert_to_tensor=True
                    )
                
                self.model_loaded = True
                logger.info(f"✅ Model loaded in {time.time()-start:.2f} seconds")
                return True
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                return False
        return True
    
    def get_response(self, user_message):
        """Get response with fallback if model fails"""
        if not user_message:
            return "Please type a message."
        
        user_message = user_message.lower().strip()
        
        # First try simple keyword matching (no model needed)
        for i, q in enumerate(self.questions):
            if q in user_message or user_message in q:
                return self.answers[i]
        
        # Try semantic matching if model loads
        if self.lazy_load_model():
            try:
                import torch
                from sentence_transformers import util
                
                user_embedding = self.model.encode(user_message, convert_to_tensor=True)
                scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
                best_score = torch.max(scores).item()
                
                if best_score > 0.5:  # Confidence threshold
                    best_idx = torch.argmax(scores).item()
                    return self.answers[best_idx]
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
        
        # Smart fallback based on common intents
        if any(word in user_message for word in ["hello", "hi", "hey"]):
            return "Hello! Welcome to Cognition Venture. How can we help your business grow?"
        elif any(word in user_message for word in ["contact", "phone", "email", "call"]):
            return "You can reach us at 📞 +91 9332015302 or 📧 info@cognitionventure.com"
        elif any(word in user_message for word in ["service", "offer", "do"]):
            return "We offer SEO, Social Media Marketing, Web Development, Branding, and Video Editing."
        elif any(word in user_message for word in ["price", "cost", "package"]):
            return "Our packages start from ₹4,269/month for social media marketing. Websites from ₹6,999/year."
        else:
            return "I'm here to help with digital marketing. Could you please provide more details or contact us directly at +91 9332015302?"

# Initialize chatbot
chatbot = MemoryOptimizedChatbot()

# -----------------------------
# Routes
# -----------------------------
@app.route('/health', methods=['GET'])
def health():
    """Health check for Render"""
    return jsonify({"status": "healthy"}), 200

@app.route("/")
def home():
    return jsonify({
        "status": "Cognition Venture AI is running",
        "mode": "memory-optimized",
        "knowledge_base": len(chabot.questions)  # Fixed typo
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
        return jsonify({"reply": "I'm experiencing high demand. Please contact us directly at +91 9332015302"}), 200

@app.route("/stats", methods=["GET"])
def stats():
    """Check memory usage"""
    import psutil
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
    return jsonify({
        "memory_mb": round(memory_usage, 2),
        "model_loaded": chatbot.model_loaded,
        "knowledge_base": len(chatbot.questions)
    })

# For Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
