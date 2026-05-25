from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")

class Request(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "AI backend running"}

@app.post("/ai")
def ai(req: Request):
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": req.prompt}
        )

        data = response.json()

        return {
            "reply": data[0]["generated_text"] if isinstance(data, list)
            else data.get("generated_text", "No response")
        }

    except Exception as e:
        return {"reply": "AI error", "error": str(e)}
