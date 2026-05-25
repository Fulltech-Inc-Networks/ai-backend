from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

app = FastAPI()

client = InferenceClient(token=os.getenv("HF_API_KEY"))

class Request(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "AI backend running"}

@app.post("/ai")
def ai(req: Request):
    try:
        response = client.text_generation(
            model="google/flan-t5-large",
            prompt=req.prompt,
            max_new_tokens=200
        )

        return {"reply": response}

    except Exception as e:
        return {"reply": "AI error", "error": str(e)}
