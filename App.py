from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.route("/")
def home():
    return "Desk Buddy backend is running"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please ask a question."}), 400

    if not OPENROUTER_API_KEY:
        return jsonify({"answer": "Missing API key on server."}), 500

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DeskBuddy, a small desk robot assistant. "
                    "Reply in very simple human-friendly language. "
                    "Keep answers short, clear, and easy to read. "
                    "Use 2 to 4 short lines maximum. "
                    "Avoid long paragraphs, markdown, and bullet points unless necessary."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "max_tokens": 100,
        "temperature": 0.6
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

    if r.status_code != 200:
        return jsonify({
            "answer": f"Error {r.status_code}",
            "details": r.text
        }), r.status_code

    result = r.json()
    answer = result["choices"][0]["message"]["content"].strip()

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)