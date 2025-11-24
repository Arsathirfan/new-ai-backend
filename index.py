from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/generate_python")
def generate_python(req: PromptRequest):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": req.prompt}]}]
    }

    response = requests.post(GEMINI_URL, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        try:
            ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"response": ai_text}
        except Exception:
            return {"error": "Unexpected Gemini API response", "raw": data}
    else:
        return {"error": "Gemini API call failed", "details": response.text}

@app.get("/api")
def root():
    return {"message": "Python FastAPI endpoint active!"}
