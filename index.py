from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Stable Model ID for the 2.5 series
MODEL_ID = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_ai_content(req: PromptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in environment")

    headers = {"Content-Type": "application/json"}
    
    # CORRECT PAYLOAD for Gemini 2.5 Flash
    # thinking_budget is an integer: 0 (off), -1 (auto), or a specific token count (e.g. 1024)
    payload = {
        "contents": [
            {
                "parts": [{"text": req.prompt}]
            }
        ],
        "generation_config": {
            "thinking_config": {
                "include_thoughts": True,  # Allows you to see the reasoning in the raw response
                "thinking_budget": 1024    # Number of tokens dedicated to 'thinking'
            },
            "temperature": 1.0,
            "max_output_tokens": 4096
        }
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload)
        
        # Error handling for API issues
        if response.status_code != 200:
            return {
                "error": f"API Error {response.status_code}", 
                "details": response.json()
            }

        data = response.json()
        
        # Parsing logic: Extracting the final response text
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            
            # The 'text' part is usually the final answer
            ai_text = ""
            for part in parts:
                if "text" in part:
                    ai_text = part["text"]
            
            return {
                "model": MODEL_ID,
                "response": ai_text
            }
        
        return {"error": "Model returned an empty response", "raw": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
def root():
    return {"message": f"FastAPI active using {MODEL_ID}"}
