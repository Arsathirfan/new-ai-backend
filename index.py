from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import json

app = FastAPI()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# The latest flagship model as of March 2026
MODEL_ID = "gemini-3-flash-preview"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_ai_content(req: PromptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set")

    headers = {"Content-Type": "application/json"}
    
    # Gemini 3 Payload Structure
    # Note: Use "LOW" for speed, "HIGH" for complex reasoning
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": req.prompt}]
            }
        ],
        "generationConfig": {
            "thinking_level": "LOW",  # Controls reasoning depth vs latency
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload)
        
        # Specific handler for Quota (429) or Bad Request (400)
        if response.status_code != 200:
            error_msg = f"API Error {response.status_code}: {response.text}"
            return {"error": "Gemini API call failed", "details": error_msg}

        data = response.json()
        
        # Safe extraction of the text response
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                ai_text = parts[0].get("text", "")
                return {
                    "model": MODEL_ID,
                    "response": ai_text
                }
        
        return {"error": "Model returned an empty response", "raw": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/test")
def root():
    return {
        "status": "online",
        "model": MODEL_ID,
        "message": "FastAPI with Gemini 3 Flash is active."
    }
