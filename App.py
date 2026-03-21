from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

OPENROUTER_API_KEY = os.getenv("Osk-or-v1-7de630728e21ac5a3b74f86fe62365effa3e0b2f2026c3549b4f1468635e5b13")

@app.route("/")
def home():
    return "Desk Buddy backend is running"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "No question received"}), 400

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

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if r.status_code != 200:
        return jsonify({
            "answer": "OpenRouter error",
            "details": r.text
        }), 500

    result = r.json()
    answer = result["choices"][0]["message"]["content"]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)