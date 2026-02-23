from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import sys
import time
import difflib
import re

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
# Intelligent Chatbot Class
# -----------------------------
class IntelligentChatbot:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.model = None
        self.question_embeddings = None
        self.model_loaded = False
        self.load_dataset()
        self.build_intent_patterns()
        
    def build_intent_patterns(self):
        """Build intent recognition patterns"""
        self.intents = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings'],
                'response': 'Hello! Welcome to Cognition Venture. How can I help your business grow today?'
            },
            'contact': {
                'patterns': ['contact', 'phone', 'call', 'whatsapp', 'email', 'reach', 'talk to', 'get in touch'],
                'response': '📞 Call: +91 9332015302\n📧 Email: info@cognitionventure.com\n💬 WhatsApp: +91 8016786669'
            },
            'services': {
                'patterns': ['service', 'offer', 'provide', 'do you do', 'can you help with', 'what do you'],
                'response': 'We offer:\n• SEO & Organic Growth\n• Social Media Marketing\n• Web Development\n• Branding & Design\n• Video Editing\n• PPC Advertising'
            },
            'seo': {
                'patterns': ['seo', 'search engine', 'ranking', 'google ranking', 'optimization'],
                'response': 'Yes, we provide professional SEO services including on-page optimization, technical SEO, local SEO, and link building. Results typically visible in 3-6 months.'
            },
            'website': {
                'patterns': ['website', 'web development', 'site', 'web design', 'ecommerce', 'e-commerce'],
                'response': 'Websites start at ₹6,999/year including hosting, domain, and mobile optimization. Professional plans at ₹14,999/year with SEO optimization.'
            },
            'pricing': {
                'patterns': ['price', 'cost', 'rate', 'package', 'pricing', 'how much', 'charges', 'fee'],
                'response': '💰 Social Media: ₹4,269/month\n🌐 Website: ₹6,999/year\n🎨 Logo: ₹2,999\n📊 SEO: Custom quote\n🎯 Google Ads: ₹5,000/month + ad spend'
            },
            'facebook_ads': {
                'patterns': ['facebook ad', 'fb ad', 'meta ad', 'facebook marketing', 'facebook promotion'],
                'response': 'Facebook ads start at ₹4,269/month (₹100/day + tax). Includes ad management, creative design, and performance tracking.'
            },
            'google_ads': {
                'patterns': ['google ad', 'google adwords', 'ppc', 'pay per click', 'google advertising'],
                'response': 'Google Ads campaigns start at ₹5,000/month + ad spend. Includes keyword research, ad creation, and monthly optimization.'
            },
            'location': {
                'patterns': ['location', 'address', 'where', 'malda', 'office', 'based'],
                'response': '📍 We are based in English Bazar, Malda, West Bengal, India. We serve clients across India and internationally.'
            },
            'experience': {
                'patterns': ['experience', 'expertise', 'background', 'worked with', 'clients'],
                'response': 'We have worked with 50+ clients across real estate, education, healthcare, retail, and startups. 95% client retention rate.'
            },
            'consultation': {
                'patterns': ['consultation', 'consult', 'discuss', 'meeting', 'call', 'strategy'],
                'response': 'Yes! We offer a free 30-minute consultation. Contact us to schedule.'
            },
            'logo': {
                'patterns': ['logo', 'branding', 'identity'],
                'response': 'Professional logo design starts at ₹2,999. Includes multiple concepts, revisions, and all file formats.'
            },
            'video': {
                'patterns': ['video', 'editing', 'reel', 'promotional video'],
                'response': 'Video editing starts at ₹2,000 per video. We create promotional videos, reels, and social media content.'
            },
            'thanks': {
                'patterns': ['thank', 'thanks', 'appreciate', 'grateful'],
                'response': "You're welcome! 😊 Let us know if you need anything else."
            },
            'bye': {
                'patterns': ['bye', 'goodbye', 'see you', 'take care'],
                'response': 'Thank you for visiting Cognition Venture. Have a great day! 👋'
            }
        }
        
    def load_dataset(self):
        """Load dataset with error handling"""
        try:
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
        for intent, data in self.intents.items():
            self.questions.append(data['patterns'][0])
            self.answers.append(data['response'])
        logger.info(f"Created {len(self.questions)} default responses")
    
    def lazy_load_model(self):
        """Load model only when needed"""
        if not self.model_loaded:
            try:
                logger.info("Loading AI model (this may take a few seconds)...")
                start = time.time()
                
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
    
    def extract_keywords(self, text):
        """Extract important keywords from text"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it',
                     'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                     'her', 'its', 'our', 'their', 'to', 'for', 'with', 'by', 'at', 'from',
                     'in', 'out', 'on', 'off', 'over', 'under', 'again', 'then', 'once',
                     'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                     'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                     'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'}
        
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def fuzzy_match(self, user_message, threshold=0.6):
        """Fuzzy string matching for similar phrases"""
        best_match = None
        best_ratio = 0
        
        for q in self.questions:
            ratio = difflib.SequenceMatcher(None, user_message, q).ratio()
            if ratio > best_ratio and ratio > threshold:
                best_ratio = ratio
                best_match = q
        
        if best_match:
            idx = self.questions.index(best_match)
            return self.answers[idx], best_ratio
        return None, 0
    
    def detect_intent(self, user_message):
        """Detect intent from user message"""
        user_message_lower = user_message.lower()
        user_keywords = self.extract_keywords(user_message_lower)
        
        best_intent = None
        best_score = 0
        
        for intent, data in self.intents.items():
            score = 0
            for pattern in data['patterns']:
                if pattern in user_message_lower:
                    score += 1
                # Check for partial matches
                pattern_words = pattern.split()
                for word in pattern_words:
                    if word in user_keywords:
                        score += 0.5
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        if best_score >= 1:
            return self.intents[best_intent]['response'], best_score
        return None, 0
    
    def get_response(self, user_message):
        """Intelligent response system"""
        if not user_message:
            return "Please type a message."
        
        user_message = user_message.lower().strip()
        logger.info(f"Processing: {user_message}")
        
        # Step 1: Try exact match (fastest)
        for i, q in enumerate(self.questions):
            if user_message == q:
                logger.info(f"Exact match found")
                return self.answers[i]
        
        # Step 2: Try contains match (if question is part of message)
        for i, q in enumerate(self.questions):
            if q in user_message:
                logger.info(f"Contains match found: {q}")
                return self.answers[i]
        
        # Step 3: Try intent detection (understands what user wants)
        intent_response, intent_score = self.detect_intent(user_message)
        if intent_response:
            logger.info(f"Intent detected with score {intent_score}")
            return intent_response
        
        # Step 4: Try fuzzy matching (handles typos and variations)
        fuzzy_response, fuzzy_score = self.fuzzy_match(user_message)
        if fuzzy_response:
            logger.info(f"Fuzzy match found with score {fuzzy_score}")
            return fuzzy_response
        
        # Step 5: Try semantic matching with AI model
        if self.lazy_load_model():
            try:
                import torch
                from sentence_transformers import util
                
                user_embedding = self.model.encode(user_message, convert_to_tensor=True)
                scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
                best_score = torch.max(scores).item()
                
                if best_score > 0.6:  # Confidence threshold
                    best_idx = torch.argmax(scores).item()
                    logger.info(f"Semantic match found with score {best_score}")
                    return self.answers[best_idx]
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
        
        # Step 6: Check for price-related queries specifically
        price_keywords = ['price', 'cost', 'rate', 'charge', 'fee', 'how much']
        if any(word in user_message for word in price_keywords):
            if 'google' in user_message and ('ad' in user_message or 'ads' in user_message):
                return "Google Ads campaigns start at ₹5,000/month + ad spend. Would you like more details?"
            elif 'facebook' in user_message and ('ad' in user_message or 'ads' in user_message):
                return "Facebook ads start at ₹4,269/month (₹100/day + tax)."
            elif 'website' in user_message:
                return "Websites start at ₹6,999/year including hosting and domain."
            elif 'seo' in user_message:
                return "SEO packages start at ₹5,000/month. Custom quote based on your needs."
            elif 'logo' in user_message:
                return "Logo design starts at ₹2,999."
        
        # Step 7: Keyword-based fallback
        keywords = self.extract_keywords(user_message)
        if keywords:
            for i, q in enumerate(self.questions):
                q_keywords = self.extract_keywords(q)
                if any(kw in q_keywords for kw in keywords):
                    logger.info(f"Keyword match found: {keywords}")
                    return self.answers[i]
        
        # Step 8: Context-aware fallback
        if len(user_message.split()) < 3:
            return "I'm here to help! Could you please provide more details about what you're looking for? You can ask about our services, pricing, or contact information."
        
        # Final fallback
        return "I want to make sure you get the right information. Could you please rephrase your question or contact us directly at +91 9332015302? Our team would be happy to help!"

# Initialize chatbot
chatbot = IntelligentChatbot()

# -----------------------------
# Routes
# -----------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@app.route("/", methods=['GET', 'HEAD'])
def home():
    try:
        if request.method == 'HEAD':
            return '', 200
        return jsonify({
            "status": "Cognition Venture AI is running",
            "mode": "intelligent",
            "knowledge_base": len(chatbot.questions)
        })
    except Exception as e:
        logger.error(f"Home endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"reply": "Invalid request format"}), 400
            
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"reply": "Please type a message."})
        
        # Get intelligent response
        reply = chatbot.get_response(user_message)
        
        return jsonify({"reply": reply})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"reply": "I'm experiencing technical issues. Please contact us directly at +91 9332015302"}), 200

@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "knowledge_base": len(chatbot.questions),
        "model_loaded": chatbot.model_loaded
    })

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({"reply": "Internal server error. Our team has been notified."}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"reply": "Endpoint not found. Try / or /chat"}), 404

# For Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
