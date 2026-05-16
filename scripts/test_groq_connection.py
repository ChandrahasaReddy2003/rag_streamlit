import os
from pathlib import Path

import httpx
import truststore
from dotenv import load_dotenv

truststore.inject_into_ssl()

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

print("API key found:", bool(api_key))
print("Model:", model)

try:
    response = httpx.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )

    print("Status code:", response.status_code)
    print(response.text[:1000])

except Exception as exc:
    print("Connection failed")
    print("Error type:", type(exc))
    print("Error:", repr(exc))
