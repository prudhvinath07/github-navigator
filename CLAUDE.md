# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
GitHub Navigator 3000 is a Flask-based web application that provides AI-powered assistance for exploring and understanding GitHub repositories. The application uses the gitingest library to extract repository content and leverages Requesty AI (Gemini models) to provide conversational assistance about codebases.

## Common Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally with default port (8080)
python app.py

# Run with custom port
python app.py --port 9090

# Run command-line interface
python github_companion.py --api-key YOUR_API_KEY
```

### Docker Commands
```bash
# Build Docker image
docker build -t github-navigator .

# Run container
docker run -p 8080:8080 github-navigator
```

### Environment Setup
Create a `.env` file with:
```
REQUESTY_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

## Architecture

### Core Components

**app.py**: Main Flask application with session management, rate limiting, and web interface endpoints
- `/` - Main chat interface
- `/chat` - POST endpoint for processing user messages
- `/reset` - POST endpoint for resetting conversations
- `/health` - Health check endpoint

**github_companion.py**: Core AI assistant logic
- `GitHubCompanion` class handles repository analysis and conversation management
- Uses gitingest library to extract repository content (summary, file tree, file contents)
- Implements token counting and content truncation for large repositories
- Manages conversation history and API interactions with Requesty AI
- Can be used standalone via command-line interface

**templates/index.html**: Single-page web interface
- Bootstrap-based responsive design with GitHub-inspired styling
- Real-time chat interface with markdown rendering
- Dark/light theme toggle with localStorage persistence
- Rate limiting handling and error display
- Syntax highlighting for code blocks

### Key Features
- **Repository Analysis**: Automatically extracts and processes GitHub repository content using gitingest
- **Token Management**: Intelligent content truncation to stay within API limits (800K tokens for conversation buffer)
- **Session Management**: Per-session conversation tracking with automatic saving
- **Rate Limiting**: 5 requests per minute per session with automatic retry logic
- **Security**: HTTP-only cookies, CSRF protection, secure session configuration

### Dependencies
- **Flask 2.3.3**: Web framework
- **gitingest**: Repository content extraction
- **openai 1.5.0**: API client for Requesty AI integration
- **tiktoken**: Token counting for content management
- **Bootstrap 5.3.0**: Frontend styling
- **marked.js**: Markdown rendering
- **highlight.js**: Code syntax highlighting

### API Integration
Uses Requesty AI (router.requesty.ai) with Google Gemini models:
- Primary model: `google/gemini-2.0-flash-thinking-exp-01-21`
- Implements retry logic for rate limiting
- Handles API errors gracefully with user-friendly messages

### Data Flow
1. User provides GitHub repository URL
2. gitingest extracts repository content (summary, tree, files)
3. Content is tokenized and truncated if needed
4. AI assistant processes user queries with repository context
5. Responses are rendered with markdown and syntax highlighting

## Development Notes
- Session data is stored in memory (sessions dict) - not persistent across restarts
- Conversation history is saved to JSON files when sessions end
- Rate limiting is per-session, not global
- The application is designed for deployment on Hugging Face Spaces with Docker