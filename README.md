# GitHub Navigator

An interactive chatbot that helps you understand and work with GitHub repositories. It extracts repository information using gitingest and answers questions about the codebase using AI models.

## Features

- Extract and analyze GitHub repositories via URL
- Multi-provider AI support: Google Gemini (primary) + OpenRouter (fallback)
- Automatic model fallback on rate limits or errors
- Token-aware context management for large repos
- Conversation history across interactions
- Web interface + command-line interface
- Production-ready with waitress WSGI server

## Prerequisites

- Python 3.10+
- At least one API key:
  - [Google AI Studio](https://aistudio.google.com/apikey) (recommended, primary)
  - [OpenRouter](https://openrouter.ai/keys) (free tier available, fallback)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd github-navigator
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key(s)
   ```

## Usage

### Web Interface

```bash
python app.py
```

Open `http://localhost:8080` in your browser and paste a GitHub repository URL to start chatting.

### Command Line

```bash
python github_companion.py
```

## Model Fallback Chain

The app tries models in this order, automatically falling back on rate limits or errors:

1. **Google Gemini** (requires `GOOGLE_API_KEY`)
   - `gemini-2.0-flash`
   - `gemini-2.0-flash-lite`

2. **OpenRouter** (requires `OPENROUTER_API_KEY`)
   - `meta-llama/llama-3.3-70b-instruct:free`
   - `qwen/qwen3-coder:free`
   - `mistralai/mistral-small-3.1-24b-instruct:free`
   - `nousresearch/hermes-3-llama-3.1-405b:free`
   - `nvidia/nemotron-3-super-120b-a12b:free`

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | One of these | Google AI Studio API key |
| `OPENROUTER_API_KEY` | is required | OpenRouter API key |
| `SECRET_KEY` | No | Flask session secret (auto-generated if unset) |
| `PORT` | No | Server port (default: 8080) |
| `FLASK_DEBUG` | No | Set `true` for Flask dev server |

## License

[MIT](LICENSE)
