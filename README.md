# Cognition Venture AI – Memory Optimized Flask Chatbot

A lightweight, memory-optimized AI chatbot API built with Flask.
Designed for deployment on low-memory environments such as Render, this chatbot supports:

* Keyword matching
* Semantic similarity search (Sentence Transformers)
* Lazy model loading
* Smart fallback responses
* Memory usage monitoring
* Production-ready logging and error handling

---

## Features

### Memory-Optimized Architecture

* Loads dataset first
* Loads AI model only when needed (lazy loading)
* Falls back to keyword matching if model fails
* Designed for low RAM servers

### Intelligent Response System

1. Keyword matching (fast, no AI model required)
2. Semantic similarity using `all-MiniLM-L6-v2`
3. Smart intent-based fallback logic

### Production Ready

* CORS enabled
* Health check endpoint
* Memory monitoring endpoint
* Proper logging
* Error handlers (404, 405, 500)
* HEAD request support for Render health checks

---

## Tech Stack

* Python 3.9+
* Flask
* Flask-CORS
* Sentence Transformers
* PyTorch
* psutil (optional, for memory stats)

---

## Project Structure

```
project/
│
├── app.py
├── dataset.txt
├── requirements.txt
└── README.md
```

---

## Dataset Format

The chatbot reads a `dataset.txt` file.

Format:

```
question ||| answer
```

Example:

```
hello ||| Hello! Welcome to Cognition Venture.
services ||| We offer SEO, Social Media Marketing, and Web Development.
```

Rules:

* Use `|||` as separator
* One Q&A pair per line
* Lines starting with `#` are ignored

If no dataset is found, default responses are automatically created.

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

Example `requirements.txt`:

```
flask
flask-cors
sentence-transformers
torch
psutil
```

---

## Running Locally

```
python app.py
```

Server runs on:

```
http://localhost:5000
```

---

## API Endpoints

### 1. Health Check

**GET** `/health`

Response:

```json
{
  "status": "healthy",
  "timestamp": 1700000000
}
```

---

### 2. Home

**GET** `/`

Response:

```json
{
  "status": "Cognition Venture AI is running",
  "mode": "memory-optimized",
  "knowledge_base": 25
}
```

---

### 3. Chat

**POST** `/chat`

Request:

```json
{
  "message": "What services do you provide?"
}
```

Response:

```json
{
  "reply": "We offer SEO & Organic Growth..."
}
```

---

### 4. Stats (Memory Monitoring)

**GET** `/stats`

Response:

```json
{
  "memory_mb": 124.5,
  "model_loaded": true,
  "knowledge_base": 25
}
```

If `psutil` is not installed, memory usage is skipped.

---

## How It Works

### Step 1 – Keyword Matching

Checks if the message directly matches stored questions.

### Step 2 – Semantic Matching

If keyword fails:

* Loads SentenceTransformer model
* Encodes user input
* Computes cosine similarity
* Returns best match if score > 0.5

### Step 3 – Smart Fallback

Handles common intents:

* Greeting
* Contact
* Services
* Pricing
* Location
* Thanks / Goodbye

---

## Deployment on Render

1. Push code to GitHub
2. Create a new Web Service on Render
3. Add:

   * Build Command:

     ```
     pip install -r requirements.txt
     ```
   * Start Command:

     ```
     python app.py
     ```
4. Render automatically assigns PORT

App already supports:

* Dynamic port binding
* HEAD health checks
* Production logging

---

## Memory Optimization Strategy

* No model loaded at startup
* Embeddings created only once
* Model loads only on first semantic request
* Fallback ensures zero-crash operation

Ideal for:

* 512MB RAM instances
* Free hosting tiers
* Lightweight SaaS backends

---

## Error Handling

* 404 – Endpoint not found
* 405 – Method not allowed
* 500 – Internal server error
* Graceful model failure fallback

---

## Example Use Case

This chatbot is designed for:

* Digital marketing agencies
* Business websites
* Lead generation systems
* WhatsApp or web chat integrations
* Low-cost AI assistants

---

## Author

Cognition Venture
Digital Marketing & AI Solutions
Malda, West Bengal, India

Contact:

* Phone: +91 9332015302
* Email: [info@cognitionventure.com](mailto:info@cognitionventure.com)

---

## License

MIT License

---

If you want, I can also generate:

* A production-grade `requirements.txt`
* A Dockerfile
* A version optimized for Railway
* A React frontend chat UI
* Or convert this into a SaaS-ready architecture
