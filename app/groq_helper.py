import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=api_key)


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

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {str(e)}"