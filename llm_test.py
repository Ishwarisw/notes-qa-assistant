import os
from dotenv import load_dotenv
from google import genai

MODEL = "gemini-flash-latest"

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("No API key found. Check your .env file.")

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model=MODEL,
        contents="In one sentence, what is RAG in AI?",
    )
    print(response.text)
except Exception as e:
    print("API call failed:", e)
