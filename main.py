from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Request(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {"status": "Groq AI running on Render"}


@app.post("/ai")
def ai(req: Request):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": req.prompt}
            ]
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "reply": "AI error",
            "error": str(e)
        }
