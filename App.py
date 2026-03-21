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
        return jsonify({"answer": "Missing API key"}), 500

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
    except Exception as e:
        return jsonify({"answer": f"Request failed: {str(e)}"}), 500

    return jsonify({
        "status_code": r.status_code,
        "raw": r.text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)