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
            
            loaded = False
            for path in paths:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and "|||" in line and not line.startswith('#'):
                                try:
                                    q, a = line.split("|||", 1)
                                    self.questions.append(q.strip().lower())
                                    self.answers.append(a.strip())
                                except:
                                    continue
                    logger.info(f"✅ Loaded {len(self.questions)} Q&A pairs from {path}")
                    loaded = True
                    return
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.error(f"Error reading {path}: {e}")
                    continue
            
            if not loaded:
                logger.warning("No dataset found, using default responses")
                self.create_default_dataset()
            
        except Exception as e:
            logger.error(f"Dataset error: {e}")
            self.create_default_dataset()
    
    def create_default_dataset(self):
        """Create default responses if no dataset"""
        defaults = [
            ("hello", "Hello! Welcome to Cognition Venture. How can we help your business grow?"),
            ("hi", "Hi there! Cognition Venture digital marketing assistant here."),
            ("contact", "Contact us: 📞 +91 9332015302 or 📧 info@cognitionventure.com"),
            ("services", "We offer SEO, Social Media Marketing, Web Development, and Branding."),
            ("price", "Packages start from ₹4,269/month. Websites from ₹6,999/year."),
            ("location", "We're located in English Bazar, Malda, West Bengal, India.")
        ]
        for q, a in defaults:
            self.questions.append(q)
            self.answers.append(a)
        logger.info(f"Created {len(self.questions)} default responses")
    
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
        if any(word in user_message for word in ["hello", "hi", "hey", "good morning"]):
            return "Hello! Welcome to Cognition Venture. How can we help your business grow today?"
        elif any(word in user_message for word in ["contact", "phone", "email", "call", "whatsapp"]):
            return "📞 Call: +91 9332015302\n📧 Email: info@cognitionventure.com\n💬 WhatsApp: +91 8016786669"
        elif any(word in user_message for word in ["service", "offer", "do", "provide"]):
            return "We offer:\n• SEO & Organic Growth\n• Social Media Marketing\n• Web Development\n• Branding & Design\n• Video Editing"
        elif any(word in user_message for word in ["price", "cost", "package", "rate"]):
            return "💰 Social Media: ₹4,269/month\n🌐 Website: ₹6,999/year\n🎨 Logo Design: ₹2,999\n📊 SEO: Custom quote"
        elif any(word in user_message for word in ["location", "where", "address", "malda"]):
            return "📍 We're based in English Bazar, Malda, West Bengal - serving clients worldwide!"
        elif any(word in user_message for word in ["thank", "thanks"]):
            return "You're welcome! 😊 Let us know if you need any help with digital growth."
        elif any(word in user_message for word in ["bye", "goodbye"]):
            return "Thank you for visiting Cognition Venture. Have a great day! 👋"
        else:
            return "I'm here to help with digital marketing. Could you please provide more details or contact us directly at +91 9332015302?"

# -----------------------------
# Initialize chatbot - FIXED: changed from 'chabot' to 'chatbot'
# -----------------------------
chatbot = MemoryOptimizedChatbot()

# -----------------------------
# Routes
# -----------------------------
@app.route('/health', methods=['GET'])
def health():
    """Health check for Render"""
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@app.route("/", methods=['GET', 'HEAD'])
def home():
    """Home endpoint - handles both GET and HEAD requests"""
    try:
        if request.method == 'HEAD':
            # Render sends HEAD requests for health checks
            return '', 200
            
        # For GET requests, return the full JSON
        return jsonify({
            "status": "Cognition Venture AI is running",
            "mode": "memory-optimized",
            "knowledge_base": len(chatbot.questions)  # FIXED: changed from 'chabot' to 'chatbot'
        })
    except Exception as e:
        logger.error(f"Home endpoint error: {e}")
        if request.method == 'HEAD':
            return '', 500
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"reply": "Invalid request format"}), 400
            
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"reply": "Please type a message."})
        
        # Get response
        reply = chatbot.get_response(user_message)  # FIXED: changed from 'chabot' to 'chatbot'
        
        return jsonify({"reply": reply})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"reply": "I'm experiencing technical issues. Please contact us directly at +91 9332015302"}), 200

@app.route("/stats", methods=["GET"])
def stats():
    """Check memory usage (optional)"""
    try:
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
        return jsonify({
            "memory_mb": round(memory_usage, 2),
            "model_loaded": chatbot.model_loaded,  # FIXED: changed from 'chabot' to 'chatbot'
            "knowledge_base": len(chatbot.questions)  # FIXED: changed from 'chabot' to 'chatbot'
        })
    except ImportError:
        return jsonify({
            "model_loaded": chatbot.model_loaded,  # FIXED: changed from 'chabot' to 'chatbot'
            "knowledge_base": len(chatbot.questions),  # FIXED: changed from 'chabot' to 'chatbot'
            "note": "psutil not installed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({"reply": "Internal server error. Our team has been notified."}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"reply": "Endpoint not found. Try / or /chat"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    logger.error(f"405 error: {error}")
    return jsonify({"reply": "Method not allowed"}), 405

# For Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
