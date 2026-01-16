import os
import sqlite3
from typing import Any, Dict, List

from google import genai
from memori import Memori
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

def get_sqlite_connection():
    """Return a connection to the local SQLite database."""
    return sqlite3.connect("memori.db")


def require_api_key(env_var: str) -> str:
    """Fetch an API key or raise a helpful error."""
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Set {env_var} in your environment before running this script.")
    return api_key


def make_client():
    provider = os.getenv("MEMORI_PROVIDER", "gemini").lower()
    if provider == "gemini":
        api_key = require_api_key("GOOGLE_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=api_key)
        # Track chosen model for reuse in demo flows.
        client._memori_model = model
        return client
    else:
        api_key = require_api_key("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=api_key)
        # attach model name for downstream use
        client._memori_model = model
        return client


def run_quickstart():
    # Initialize client and wire Memori to SQLite.
    client = make_client()
    memori = Memori(conn=get_sqlite_connection).llm.register(client)
    memori.attribution(entity_id="123456", process_id="test-ai-agent")
    memori.config.storage.build()

    # Store a fact.
    response = (
        client.models.generate_content(
            model=getattr(client, "_memori_model", None) or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=[{"role": "user", "parts": [{"text": "My favorite color is blue."}]}],
        )
        if hasattr(client, "models")
        else client.chat.completions.create(
            model=getattr(client, "_memori_model", None) or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": "My favorite color is blue."}],
        )
    )
    print((getattr(response, "text", None) or response.choices[0].message.content) + "\n")

    # Ensure augmentation finishes for this short-lived process.
    memori.augmentation.wait()

    # Start fresh client + Memori to show persisted memory.
    client = make_client()
    memori = Memori(conn=get_sqlite_connection).llm.register(client)
    memori.attribution(entity_id="123456", process_id="test-ai-agent")

    # Recall the stored fact.
    response = (
        client.models.generate_content(
            model=getattr(client, "_memori_model", None) or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=[{"role": "user", "parts": [{"text": "What's my favorite color?"}]}],
        )
        if hasattr(client, "models")
        else client.chat.completions.create(
            model=getattr(client, "_memori_model", None) or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": "What's my favorite color?"}],
        )
    )
    print((getattr(response, "text", None) or response.choices[0].message.content) + "\n")


if __name__ == "__main__":
    run_quickstart()

