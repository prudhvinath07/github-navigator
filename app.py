# python app.py --port 9090

from flask import Flask, render_template, request, jsonify, session
import os
import sys
import argparse
from github_companion import GitHubCompanion
from dotenv import load_dotenv
import uuid  # Using Python's built-in uuid module
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# Store active sessions
sessions = {}

# Rate limiting
request_timestamps = {}
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 5

@app.route("/")
def index():
    """Render the main page"""
    # Generate a unique session ID if one doesn't exist
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    return render_template("index.html")

def is_rate_limited(session_id):
    """Check if the session is rate limited"""
    current_time = time.time()

    # Initialize timestamps for this session if not exists
    if session_id not in request_timestamps:
        request_timestamps[session_id] = []

    # Remove timestamps outside the window
    request_timestamps[session_id] = [
        ts for ts in request_timestamps[session_id]
        if ts > current_time - RATE_LIMIT_WINDOW
    ]

    # Check if too many requests in the window
    if len(request_timestamps[session_id]) >= MAX_REQUESTS_PER_WINDOW:
        return True

    # Add current timestamp
    request_timestamps[session_id].append(current_time)
    return False

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests"""
    data = request.json
    message = data.get("message", "")
    session_id = session.get("session_id")

    if not session_id:
        return jsonify({"error": "No valid session"}), 400

    # Check rate limiting
    if is_rate_limited(session_id):
        return jsonify({
            "response": "You're sending requests too quickly. Please wait a moment before trying again.",
            "token_count": 0,
            "rate_limited": True
        })

    try:
        # Initialize companion if not exists for this session
        if session_id not in sessions:
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            google_api_key = os.environ.get("GOOGLE_API_KEY")

            if openrouter_api_key:
                logger.info(f"OpenRouter key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:]}")
            if google_api_key:
                logger.info(f"Google key: {google_api_key[:5]}...{google_api_key[-5:]}")

            if not openrouter_api_key and not google_api_key:
                return jsonify({"error": "No API keys configured. Set OPENROUTER_API_KEY or GOOGLE_API_KEY in .env"}), 500

            sessions[session_id] = GitHubCompanion(
                openrouter_api_key=openrouter_api_key,
                google_api_key=google_api_key,
            )

        companion = sessions[session_id]
        response = companion.chat(message)

        # Check if the response contains an error message about rate limiting
        rate_limited = "rate limit" in response.lower() or "quota" in response.lower()

        return jsonify({
            "response": response,
            "token_count": companion.token_count if companion.repo_info else 0,
            "rate_limited": rate_limited
        })
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/reset", methods=["POST"])
def reset():
    """Reset the conversation"""
    session_id = session.get("session_id")

    if session_id and session_id in sessions:
        try:
            # Save conversation before resetting
            sessions[session_id].save_conversation(f"conversation_{session_id}.json")
            # Remove the session
            del sessions[session_id]
        except Exception as e:
            logger.error(f"Error in reset: {str(e)}")

    # Create new session ID
    session["session_id"] = str(uuid.uuid4())

    return jsonify({"status": "success", "message": "Conversation reset"})

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GitHub Navigator Web App")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    args = parser.parse_args()

    # Ensure the templates directory exists
    os.makedirs("templates", exist_ok=True)

    # Priority: 1. Command line argument, 2. Environment variable, 3. Default (8080)
    port = args.port if args.port else int(os.environ.get("PORT", 8080))

    logger.info(f"Starting GitHub Navigator on port {port}")

    # Run the app
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    if debug:
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        from waitress import serve
        logger.info(f"Serving with waitress on http://0.0.0.0:{port}")
        serve(app, host="0.0.0.0", port=port)
