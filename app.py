from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import re
from collections import deque
import logging

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Create Flask App
# -----------------------------
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Required for session
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------
# Load Dataset with Preprocessing
# -----------------------------
class ChatbotBrain:
    def __init__(self, dataset_path="dataset.txt"):
        self.questions = []
        self.answers = []
        self.keywords_cache = {}
        self.conversation_history = deque(maxlen=5)  # Store last 5 exchanges
        self.load_dataset(dataset_path)
        self.load_model()
        self.create_fallback_responses()
        
    def preprocess_text(self, text):
        """Clean and normalize text"""
        text = text.lower().strip()
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s?.!,]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def extract_keywords(self, text):
        """Extract important keywords from text"""
        # Remove common stop words (you can expand this list)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it',
                     'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                     'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours',
                     'theirs', 'to', 'for', 'with', 'by', 'at', 'from', 'in', 'out', 'on',
                     'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                     'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                     'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                     'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                     'will', 'just', 'don', 'should', 'now'}
        
        words = text.split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def load_dataset(self, dataset_path):
        """Load and preprocess dataset"""
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if "|||" in line:
                        try:
                            q, a = line.split("|||", 1)  # Split only first occurrence
                            processed_q = self.preprocess_text(q)
                            self.questions.append(processed_q)
                            self.answers.append(a.strip())
                            
                            # Cache keywords for each question
                            self.keywords_cache[processed_q] = self.extract_keywords(processed_q)
                        except Exception as e:
                            logger.warning(f"Error parsing line {line_num}: {e}")
                            continue
            
            logger.info(f"Loaded {len(self.questions)} question-answer pairs")
            
        except FileNotFoundError:
            logger.error(f"Dataset file {dataset_path} not found. Creating default dataset.")
            self.create_default_dataset()
    
    def create_default_dataset(self):
        """Create a default dataset if none exists"""
        default_pairs = [
            "hello|||Hello! How can I assist you today?",
            "how are you|||I'm functioning optimally, thank you for asking!",
            "what is your name|||I'm Cognition Venture Intelligent AI, your helpful assistant.",
            "who created you|||I was created by the team at Cognition Venture.",
            "what can you do|||I can answer questions, have conversations, and help with various tasks using my knowledge base.",
            "bye|||Goodbye! Feel free to return if you have more questions.",
            "thank you|||You're welcome! Is there anything else I can help with?",
            "help|||I'm here to assist you. Feel free to ask me any questions."
        ]
        
        for pair in default_pairs:
            q, a = pair.split("|||")
            processed_q = self.preprocess_text(q)
            self.questions.append(processed_q)
            self.answers.append(a)
            self.keywords_cache[processed_q] = self.extract_keywords(processed_q)
    
    def create_fallback_responses(self):
        """Create intelligent fallback responses"""
        self.fallback_responses = [
            "I'm not entirely sure about that. Could you rephrase your question?",
            "That's an interesting question. Let me think... Could you provide more details?",
            "I don't have enough information to answer that accurately. Would you like to ask something else?",
            "I'm still learning about that topic. Could you try asking in a different way?",
            "That's outside my current knowledge base. Is there something specific you'd like to know?",
            "I understand you're asking something, but I need a bit more context. Can you elaborate?"
        ]
        
        # Specific intent-based fallbacks
        self.intent_fallbacks = {
            'greeting': "Hello! How can I help you today?",
            'farewell': "Goodbye! It was nice chatting with you.",
            'thanks': "You're welcome! Happy to help!",
            'help': "I'm here to assist. What would you like to know?",
            'weather': "I don't have access to real-time weather data. You might want to check a weather service.",
            'time': "I don't have access to the current time. Please check your device.",
            'joke': "Why don't scientists trust atoms? Because they make up everything!",
            'name': "I'm Cognition Venture Intelligent AI, but you can call me CV AI for short."
        }
    
    def detect_intent(self, text):
        """Detect basic intent from user message"""
        text_lower = text.lower()
        
        intent_patterns = {
            'greeting': r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b',
            'farewell': r'\b(bye|goodbye|see you|farewell|take care)\b',
            'thanks': r'\b(thank|thanks|appreciate|grateful)\b',
            'help': r'\b(help|assist|support|guide)\b',
            'weather': r'\b(weather|temperature|rain|sunny|cloudy|forecast)\b',
            'time': r'\b(time|clock|hour|minute)\b',
            'joke': r'\b(joke|funny|laugh|humor)\b',
            'name': r'\b(name|call you|who are you)\b'
        }
        
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, text_lower):
                return intent
        return None
    
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            # Create embeddings for all questions
            if self.questions:
                self.question_embeddings = self.model.encode(
                    self.questions,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                logger.info("Model loaded and embeddings created successfully")
            else:
                logger.warning("No questions to create embeddings for")
                self.question_embeddings = None
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def find_best_match(self, user_message, threshold=0.5):
        """Find the best matching question with confidence score"""
        if not self.questions or self.question_embeddings is None:
            return None, 0.0
        
        # Create embedding for user message
        user_embedding = self.model.encode(user_message, convert_to_tensor=True)
        
        # Calculate similarities
        scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
        
        # Get top 3 matches for better analysis
        top_scores, top_indices = torch.topk(scores, min(3, len(scores)))
        
        # Check if the best match meets threshold
        best_score = top_scores[0].item()
        
        if best_score >= threshold:
            best_index = top_indices[0].item()
            return self.answers[best_index], best_score
        else:
            # Check if we have multiple decent matches
            if len(top_scores) > 1 and top_scores[1].item() > 0.4:
                # Combine insights from multiple matches (advanced feature)
                return None, best_score
            return None, best_score
    
    def get_keyword_match(self, user_message):
        """Fallback matching using keywords"""
        user_keywords = set(self.extract_keywords(user_message))
        if not user_keywords:
            return None
        
        best_match = None
        best_score = 0
        
        for q, keywords in self.keywords_cache.items():
            if keywords:
                # Calculate Jaccard similarity
                common = len(user_keywords.intersection(set(keywords)))
                if common > 0:
                    # Weight by importance of keywords
                    score = common / max(len(user_keywords), len(keywords))
                    if score > best_score and score > 0.3:  # Threshold for keyword match
                        best_score = score
                        best_match = self.answers[self.questions.index(q)]
        
        return best_match, best_score
    
    def get_response(self, user_message):
        """Main method to get response for user message"""
        # Preprocess user message
        processed_message = self.preprocess_text(user_message)
        
        # Detect intent for specialized responses
        intent = self.detect_intent(processed_message)
        
        # Check for intent-based response first
        if intent in self.intent_fallbacks and processed_message in ['hello', 'hi', 'hey']:
            return {
                'reply': self.intent_fallbacks[intent],
                'confidence': 0.9,
                'method': 'intent_match'
            }
        
        # Try semantic matching
        best_answer, confidence = self.find_best_match(processed_message)
        
        if best_answer and confidence >= 0.6:
            response = best_answer
            method = 'semantic_match'
        else:
            # Try keyword matching as fallback
            keyword_answer, keyword_confidence = self.get_keyword_match(processed_message)
            
            if keyword_answer and keyword_confidence >= 0.4:
                response = keyword_answer
                method = 'keyword_match'
                confidence = keyword_confidence
            else:
                # Use intent-based or generic fallback
                if intent in self.intent_fallbacks:
                    response = self.intent_fallbacks[intent]
                    method = 'intent_fallback'
                    confidence = 0.5
                else:
                    # Choose most appropriate fallback based on message length
                    if len(processed_message.split()) < 3:
                        response = "I'm here to help. Could you please provide more details?"
                    else:
                        response = np.random.choice(self.fallback_responses)
                    method = 'fallback'
                    confidence = 0.3
        
        # Update conversation history
        self.conversation_history.append({
            'user': user_message,
            'bot': response,
            'confidence': confidence,
            'method': method
        })
        
        return {
            'reply': response,
            'confidence': confidence,
            'method': method
        }

# Initialize the chatbot brain
brain = ChatbotBrain()

# -----------------------------
# Health Check Route
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "name": "Cognition Venture Intelligent AI",
        "version": "2.0",
        "capabilities": ["semantic_matching", "intent_detection", "context_awareness"],
        "stats": {
            "knowledge_base_size": len(brain.questions),
            "model": "all-MiniLM-L6-v2"
        }
    })

# -----------------------------
# Chat Route
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({
                "reply": "Please type a message.",
                "error": "empty_message"
            })
        
        # Get response from brain
        response_data = brain.get_response(user_message)
        
        # Prepare response with metadata
        return jsonify({
            "reply": response_data['reply'],
            "confidence": response_data['confidence'],
            "method": response_data['method'],
            "suggestions": get_suggestions(user_message) if response_data['confidence'] < 0.5 else []
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "reply": "I encountered an error processing your request. Please try again.",
            "error": str(e)
        }), 500

# -----------------------------
# Suggestions Helper
# -----------------------------
def get_suggestions(user_message):
    """Provide suggested questions based on user input"""
    common_questions = [
        "What can you do?",
        "How are you?",
        "What is your name?",
        "Tell me a joke",
        "Help me"
    ]
    
    # Return suggestions that might be relevant
    return common_questions[:3]  # Return top 3 suggestions

# -----------------------------
# Context Route (optional - for maintaining conversation state)
# -----------------------------
@app.route("/context", methods=["GET"])
def get_context():
    """Get recent conversation history"""
    return jsonify({
        "history": list(brain.conversation_history)
    })

# -----------------------------
# Train Route (optional - for updating knowledge)
# -----------------------------
@app.route("/train", methods=["POST"])
def train():
    """Add new knowledge to the bot"""
    try:
        data = request.get_json()
        new_question = data.get("question", "").strip()
        new_answer = data.get("answer", "").strip()
        
        if new_question and new_answer:
            # Add to dataset (in memory)
            processed_q = brain.preprocess_text(new_question)
            brain.questions.append(processed_q)
            brain.answers.append(new_answer)
            brain.keywords_cache[processed_q] = brain.extract_keywords(processed_q)
            
            # Update embeddings
            new_embedding = brain.model.encode([processed_q], convert_to_tensor=True)
            if brain.question_embeddings is not None:
                brain.question_embeddings = torch.cat([brain.question_embeddings, new_embedding])
            else:
                brain.question_embeddings = new_embedding
            
            # Optionally save to file
            with open("dataset.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{new_question}|||{new_answer}")
            
            return jsonify({
                "status": "success",
                "message": "New knowledge added successfully",
                "total_questions": len(brain.questions)
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Please provide both question and answer"
            }), 400
            
    except Exception as e:
        logger.error(f"Error in train endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
