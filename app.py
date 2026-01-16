import os
import sqlite3
import uuid
from typing import Any, Dict, List
from flask import Flask, render_template, request, jsonify, session
from memori import Memori
from openai import OpenAI
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "memori-labs-demo-secret-key-change-in-production")


def get_sqlite_connection():
    """Return a connection to the local SQLite database."""
    return sqlite3.connect("memori.db")


def require_api_key(env_var: str) -> str:
    """Fetch an API key or raise a helpful error."""
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Set {env_var} in your environment before running this script.")
    return api_key


def make_client(provider="openai", model_override=None):
    """Create and return the appropriate LLM client, honoring UI-selected model."""
    provider = provider.lower()
    if provider == "gemini":
        api_key = require_api_key("GOOGLE_API_KEY")
        model = model_override or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        # FIXED: Use genai.Client() for Memori compatibility
        client = genai.Client(api_key=api_key)
        client._memori_model = model
        return client
    else:
        api_key = require_api_key("OPENAI_API_KEY")
        model = model_override or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=api_key)
        client._memori_model = model
        return client


def get_or_create_session_id():
    """Get or create a unique session ID for this user."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def init_memori_for_session(provider="openai", model_override=None):
    """Initialize Memori with session-specific entity_id."""
    client = make_client(provider, model_override)
    memori = Memori(conn=get_sqlite_connection).llm.register(client)
    session_id = get_or_create_session_id()
    # Use session_id as entity_id so each user has isolated memory
    memori.attribution(entity_id=session_id, process_id="web-demo")
    memori.config.storage.build()
    model = getattr(client, "_memori_model", None) or (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") if provider == "openai" else os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    return client, memori, model


def init_client_without_memori(provider="openai", model_override=None):
    """Initialize client without Memori for comparison."""
    client = make_client(provider, model_override)
    model = getattr(client, "_memori_model", None) or (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") if provider == "openai" else os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    return client, model


@app.route('/')
def index():
    """Main chat interface."""
    return render_template('index.html')


@app.route('/theory')
def theory():
    """Theory and documentation page."""
    return render_template('theory.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with optional memory persistence."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        use_memori = data.get('use_memori', True)
        provider = data.get('provider', 'openai')
        model_override = data.get('model')
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if use_memori:
            # Initialize Memori for this session
            client, memori, model = init_memori_for_session(provider, model_override)
            
            # Create chat completion with memory
            if provider == "gemini":
                response = client.models.generate_content(
                    model=model,
                    contents=user_message
                )
                ai_response = response.text
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_message}],
                )
                ai_response = response.choices[0].message.content
            
            # Ensure augmentation finishes (memory is persisted)
            memori.augmentation.wait()
        else:
            # Use client without Memori for comparison
            client, model = init_client_without_memori(provider, model_override)
            
            # Create chat completion without memory
            if provider == "gemini":
                response = client.models.generate_content(
                    model=model,
                    contents=user_message
                )
            
                ai_response = response.text

            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_message}],
                )
                ai_response = response.choices[0].message.content
        
        return jsonify({
            'response': ai_response,
            'session_id': get_or_create_session_id(),
            'use_memori': use_memori,
            'provider': provider,
            'model': model
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the session to start fresh."""
    session.clear()
    return jsonify({'message': 'Session reset successfully'})


if __name__ == '__main__':
    # Check for at least one API key
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))
    
    if not has_openai and not has_gemini:
        print("Error: Please set at least one API key:")
        print("  - OPENAI_API_KEY for OpenAI")
        print("  - GOOGLE_API_KEY for Gemini")
        print("You can set these in your .env file or environment variables")
        exit(1)
    
    port = int(os.getenv("PORT", 5001))
    print(f"Starting Memori Labs demo on http://localhost:{port}")
    print(f"OpenAI available: {has_openai}")
    print(f"Gemini available: {has_gemini}")
    app.run(debug=True, host='0.0.0.0', port=port)
