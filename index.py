from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_ai_content(req: PromptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="API Key not found")

    payload = {"contents": [{"parts": [{"text": req.prompt}]}]}

    try:
        response = requests.post(GEMINI_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"response": ai_text}
        else:
            return {"error": "API Error", "details": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_status():
    """
    Identifies if the server is online and 
    checks if the API key is configured.
    """
    return {
        "status": "online",
        "server_info": "FastAPI is running",
        "model_active": MODEL_ID,
        "api_key_configured": GEMINI_API_KEY is not None,
        "endpoint": "/generate"
    }

@app.get("/")
def root():
    return {"message": "Server is LIVE. Use /status to check health or /generate to prompt."}
