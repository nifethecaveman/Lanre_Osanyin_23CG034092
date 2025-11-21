#!/usr/bin/env python3
"""
CLI Q&A using OpenRouter Grok model
"""

import os, re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")

if not OPENROUTER_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set in .env")

# Initialize OpenRouter client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

def preprocess(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def ask_openrouter(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning": {"enabled": True}}
    )
    return response.choices[0].message.content

def main():
    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered.")
        return
    processed = preprocess(question)
    print("\n[Processed Question]:", processed)
    print("\n[Asking OpenRouter Grok API...]")
    try:
        answer = ask_openrouter(processed)
        print("\n[Answer]:", answer)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
