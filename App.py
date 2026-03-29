from flask import Flask, request, jsonify
import requests
import os
import base64
import tempfile
import traceback
import whisper
from TTS.api import TTS

app = Flask(__name__)

# ---------------- CONFIG ----------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are DeskBuddy, a small desk robot assistant. "
    "Reply in very simple human-friendly language. "
    "Keep answers short, clear, and easy to read. "
    "Use 2 to 4 short lines maximum. "
    "Do not use markdown, bullet points, or long paragraphs."
)

# ---------------- LOAD MODELS ONCE ----------------

print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")   # use tiny for lower RAM usage

print("Loading Coqui TTS model...")
tts_model = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC",
    progress_bar=False
)

print("Models loaded successfully.")

# ---------------- HELPERS ----------------

def ask_llm(question: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY on server.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "max_tokens": 120,
        "temperature": 0.6
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: {response.text}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def speech_to_text(input_wav_path: str) -> str:
    result = whisper_model.transcribe(input_wav_path)
    text = result.get("text", "").strip()
    return text


def text_to_speech(text: str, output_wav_path: str):
    tts_model.tts_to_file(text=text, file_path=output_wav_path)


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return "Desk Buddy voice backend is running"


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Please ask a question."}), 400

        answer = ask_llm(question)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voice", methods=["POST"])
def voice():
    input_path = None
    output_path = None

    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400

        audio_file = request.files["audio"]

        if not audio_file or audio_file.filename == "":
            return jsonify({"error": "Empty audio file."}), 400

        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as in_file:
            input_path = in_file.name
            audio_file.save(input_path)

        # Step 1: Speech to text
        transcript = speech_to_text(input_path)

        if not transcript:
            return jsonify({"error": "Could not understand audio."}), 400

        # Step 2: Ask AI
        answer = ask_llm(transcript)

        # Step 3: Text to speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_file:
            output_path = out_file.name

        text_to_speech(answer, output_path)

        # Step 4: Read output audio and encode as base64
        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return jsonify({
            "transcript": transcript,
            "answer": answer,
            "audio": audio_b64
        })

    except Exception as e:
        print("VOICE ROUTE ERROR:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass

        try:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass


# ---------------- MAIN ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)