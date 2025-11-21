from flask import Flask, render_template, request, jsonify
import os, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")

if not OPENROUTER_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set in .env")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

app = Flask(__name__, template_folder="templates", static_folder="static")

def preprocess(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"success": False, "error": "No question provided"}), 400
    processed = preprocess(question)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": processed}],
            extra_body={"reasoning": {"enabled": True}}
        )
        answer = response.choices[0].message.content
        return jsonify({"success": True, "processed": processed, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
