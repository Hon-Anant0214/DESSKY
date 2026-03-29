from flask import Flask, request, jsonify
import requests
import os
import base64
import whisper
from TTS.api import TTS

app = Flask(__name__)

# API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load models once when server starts
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading TTS model...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

SYSTEM_PROMPT = (
    "You are DeskBuddy, a small desk robot assistant. "
    "Reply in simple language and keep answers short."
)

def ask_llm(question):

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

    r = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=30
    )

    if r.status_code != 200:
        raise Exception(f"OpenRouter error: {r.text}")

    result = r.json()

    return result["choices"][0]["message"]["content"].strip()


@app.route("/")
def home():
    return "Desk Buddy backend running"


@app.route("/voice", methods=["POST"])
def voice():

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    input_audio = "input.wav"
    output_audio = "response.wav"

    audio_file.save(input_audio)

    print("Transcribing audio...")

    result = whisper_model.transcribe(input_audio)

    transcript = result["text"].strip()

    print("User said:", transcript)

    answer = ask_llm(transcript)

    print("AI answer:", answer)

    print("Generating speech...")

    tts.tts_to_file(
        text=answer,
        file_path=output_audio
    )

    with open(output_audio, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode()

    return jsonify({
        "transcript": transcript,
        "answer": answer,
        "audio": audio_b64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)