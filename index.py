from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# UPDATED: Using the brand new Gemini 3 Flash Preview
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_python(req: PromptRequest):
    if not GEMINI_API_KEY:
        return {"error": "API Key is missing"}

    headers = {"Content-Type": "application/json"}
    
    # Gemini 3 supports "thinking_level" to control speed vs reasoning depth
    payload = {
        "contents": [{"parts": [{"text": req.prompt}]}],
        "generationConfig": {
            "thinking_level": "LOW" # Use "LOW" for that signature Flash speed
        }
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            # The response structure remains compatible with previous versions
            candidates = data.get("candidates", [])
            if candidates:
                ai_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return {"response": ai_text}
            return {"error": "Empty response from Gemini 3"}
        
        return {"error": f"Error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "Server Error", "details": str(e)}

@app.get("/test")
def root():
    return {"message": "FastAPI running on Gemini 3 Flash!"}
