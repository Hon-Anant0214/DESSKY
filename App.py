from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

OPENROUTER_API_KEY = os.getenv("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")

@app.route("/")
def home():
    return "Desk Buddy backend is running"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "No question received"}), 400

    if not OPENROUTER_API_KEY:
        return jsonify({"answer": "Missing OPENROUTER_API_KEY on server"}), 500

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are Desk Buddy, a helpful robot assistant."},
            {"role": "user", "content": question}
        ]
    }

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
    except Exception as e:
        return jsonify({"answer": f"Request failed: {str(e)}"}), 500

    if r.status_code != 200:
        return jsonify({
            "answer": f"OpenRouter {r.status_code}",
            "details": r.text
        }), r.status_code

    result = r.json()
    answer = result["choices"][0]["message"]["content"]
    return jsonify({"answer": answer})