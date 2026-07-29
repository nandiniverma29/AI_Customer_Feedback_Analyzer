import os
from dotenv import load_dotenv
from google import genai
load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
def analyze_review(review, sentiment):
    prompt = f"""
You are an AI Customer Feedback Analyst.

Customer Review:
{review}

Machine Learning Prediction:
{sentiment}

Provide the output in the following format:

## Review Summary

## Positive Points

## Negative Points

## Suggested Improvements

## Suggested Company Reply

Keep the answer concise, professional, and easy to understand.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text