# import os
# from pathlib import Path

# import truststore
# from dotenv import load_dotenv
# from groq import Groq

# # Corporate Windows laptops often require system certificate store injection.
# truststore.inject_into_ssl()

# ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# ENV_PATH = ROOT_DIR / ".env"

# load_dotenv(dotenv_path=ENV_PATH)


# class GroqLLMClient:
#     def __init__(self):
#         api_key = os.getenv("GROQ_API_KEY")
#         self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

#         if not api_key:
#             raise ValueError(f"GROQ_API_KEY not found. Expected .env at: {ENV_PATH}")

#         self.client = Groq(api_key=api_key)

#     def generate(self, messages, temperature=0.1, max_tokens=700):
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#             )
#             return response.choices[0].message.content

#         except Exception as exc:
#             raise RuntimeError(
#                 f"Groq connection/generation failed. "
#                 f"Model used: {self.model}. "
#                 f"Error type: {type(exc).__name__}. "
#                 f"Error detail: {repr(exc)}"
#             )

import os
from pathlib import Path

import truststore
from dotenv import load_dotenv
from groq import Groq

truststore.inject_into_ssl()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def get_secret_value(key: str, default=None):
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default


class GroqLLMClient:
    def __init__(self):
        api_key = get_secret_value("GROQ_API_KEY")
        self.model = get_secret_value("GROQ_MODEL", "llama-3.1-8b-instant")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it in .env locally or Streamlit secrets after deployment."
            )

        self.client = Groq(api_key=api_key)

    def generate(self, messages, temperature=0.1, max_tokens=700):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except Exception as exc:
            raise RuntimeError(
                f"Groq connection/generation failed. "
                f"Model used: {self.model}. "
                f"Error type: {type(exc).__name__}. "
                f"Error detail: {repr(exc)}"
            )