import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CommentRequest(BaseModel):
    comment: str


class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
        completion = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "system",
            "content": """You are a strict sentiment analysis API.

Classification rules:

- POSITIVE → Clear praise, strong satisfaction, excitement.
  Rating: 4 or 5

- NEUTRAL → Factual, mild, no strong emotion, "okay", "as expected".
  Rating: 3

- NEGATIVE → Clear dissatisfaction, complaint, disappointment.
  Rating: 1 or 2

Return ONLY valid JSON in this exact format:
{
  "sentiment": "positive | negative | neutral",
  "rating": integer
}

No explanation. No extra text."""
        },
        {
            "role": "user",
            "content": request.comment
        }
    ],
    temperature=0
)

        content = completion.choices[0].message.content

        # Convert string JSON to dictionary
        parsed = json.loads(content)

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))