# python app.py --port 9090

import os
import time
import uuid
import logging
import argparse
import threading

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

load_dotenv()

from github_companion import GitHubCompanion, DEFAULT_MODEL  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL = os.environ.get("LLM_MODEL") or DEFAULT_MODEL
REASONING = (os.environ.get("LLM_REASONING") or "").strip().lower() or None

# Sessions live in memory, keyed by an ID the browser keeps in sessionStorage
# and sends as X-Session-ID. (Cookies don't survive the cross-site iframe that
# Hugging Face Spaces embeds the app in.)
SESSION_TTL = 60 * 60  # seconds of inactivity before a session is dropped
MAX_SESSIONS = 50
MAX_MESSAGE_LENGTH = 4000

sessions = {}  # session_id -> {"companion", "last_used", "lock", "timestamps"}
sessions_lock = threading.Lock()

RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 5


def get_session_id():
    """Return the validated session ID from the request header, or None."""
    try:
        return str(uuid.UUID(request.headers.get("X-Session-ID", "")))
    except ValueError:
        return None


def get_session(session_id):
    """Fetch or create the session entry, evicting idle or excess sessions."""
    now = time.time()
    with sessions_lock:
        for sid in [s for s, entry in sessions.items() if now - entry["last_used"] > SESSION_TTL]:
            del sessions[sid]
        if session_id not in sessions:
            while len(sessions) >= MAX_SESSIONS:
                del sessions[min(sessions, key=lambda s: sessions[s]["last_used"])]
            sessions[session_id] = {"companion": None, "last_used": now,
                                    "lock": threading.Lock(), "timestamps": []}
        entry = sessions[session_id]
        entry["last_used"] = now
        return entry


def is_rate_limited(entry):
    """Check if the session is rate limited"""
    now = time.time()
    entry["timestamps"] = [ts for ts in entry["timestamps"] if ts > now - RATE_LIMIT_WINDOW]
    if len(entry["timestamps"]) >= MAX_REQUESTS_PER_WINDOW:
        return True
    entry["timestamps"].append(now)
    return False


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html", model=MODEL, reasoning=REASONING)


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests"""
    session_id = get_session_id()
    if not session_id:
        return jsonify({"error": "Missing or invalid session ID. Please reload the page."}), 400

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Message is empty."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."}), 400

    entry = get_session(session_id)
    if is_rate_limited(entry):
        return jsonify({
            "response": "You're sending requests too quickly. Please wait a moment before trying again.",
            "token_count": 0,
            "rate_limited": True,
        })

    if not entry["lock"].acquire(blocking=False):
        return jsonify({"error": "Still working on your previous message. Please wait."}), 429
    try:
        if entry["companion"] is None:
            entry["companion"] = GitHubCompanion()
        companion = entry["companion"]
        result = companion.chat(message)
        return jsonify({
            "response": result["response"],
            "reasoning": result["reasoning"],
            "token_count": companion.token_count,
            "rate_limited": result["rate_limited"],
        })
    except Exception:
        logger.exception("Error in chat")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500
    finally:
        entry["lock"].release()


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the conversation"""
    session_id = get_session_id()
    if session_id:
        with sessions_lock:
            sessions.pop(session_id, None)
    return jsonify({"status": "success", "message": "Conversation reset"})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model": MODEL, "reasoning": REASONING or "default"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub Navigator Web App")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    args = parser.parse_args()

    # Priority: 1. Command line argument, 2. Environment variable, 3. Default (8080)
    port = args.port or int(os.environ.get("PORT", 8080))
    logger.info(f"Starting GitHub Navigator on port {port} with model {MODEL}")
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1", threaded=True)
