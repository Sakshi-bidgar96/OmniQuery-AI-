
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

if not api_key:
    print("ERROR: GEMINI_API_KEY missing in .env")
else:
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60000)
        )

        print("Testing model:", model)
        print("Please wait...")

        response = client.models.generate_content(
            model=model,
            contents="Explain DBMS in simple English with one example."
        )

        print("\nGEMINI ANSWER:\n")
        print(response.text or "Empty response")

    except Exception as e:
        print("\nGEMINI ERROR:")
        print(type(e).__name__)
        print(str(e))
