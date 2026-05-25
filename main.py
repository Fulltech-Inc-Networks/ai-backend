from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

app = FastAPI()

# Get API key from Render environment variables
HF_API_KEY = os.getenv("HF_API_KEY")

# Hugging Face client
client = InferenceClient(token=HF_API_KEY)

# Request body structure
class Request(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {"status": "AI backend running"}


@app.post("/ai")
def ai(req: Request):
    try:
        # Call Hugging Face model
        response = client.text_generation(
            model="google/flan-t5-small",
            prompt=req.prompt,
            max_new_tokens=200
        )

        return {"reply": response}

    except Exception as e:
        print("ERROR:", str(e))  # shows in Render logs
        return {
            "reply": "AI error",
            "error": str(e)
        }
