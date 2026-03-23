import os
from openai import OpenAI
from google import genai
import tiktoken
import re
from gitingest import ingest
import json
import datetime
import logging
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubCompanion:
    def __init__(self, openrouter_api_key=None, google_api_key=None):
        """Initialize the GitHub Companion chatbot"""
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")

        if not self.openrouter_api_key and not self.google_api_key:
            raise ValueError("At least one API key is required (OPENROUTER_API_KEY or GOOGLE_API_KEY)")

        # Get OpenRouter site configuration
        self.site_url = os.environ.get("OPENROUTER_SITE_URL", "https://github-navigator.local")
        self.site_name = os.environ.get("OPENROUTER_SITE_NAME", "GitHub Navigator")

        # Initialize OpenRouter client
        self.openrouter_client = None
        if self.openrouter_api_key:
            try:
                api_key_preview = f"{self.openrouter_api_key[:5]}...{self.openrouter_api_key[-5:]}"
                logger.info(f"Initializing OpenRouter with key: {api_key_preview}")
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_api_key
                )
                logger.info("OpenRouter client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenRouter client: {e}")

        # Initialize Google Gemini client
        self.google_client = None
        if self.google_api_key:
            try:
                api_key_preview = f"{self.google_api_key[:5]}...{self.google_api_key[-5:]}"
                logger.info(f"Initializing Google Gemini with key: {api_key_preview}")
                self.google_client = genai.Client(api_key=self.google_api_key)
                logger.info("Google Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Google Gemini client: {e}")

        # Fallback chain: OpenRouter free models + Google Gemini models
        self.openrouter_models = [
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-coder:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ]
        self.google_models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
        self.model = self.google_models[0] if self.google_client else self.openrouter_models[0]
        self.conversation_history = []
        self.repo_info = None
        self.token_count = 0
        # Gemini 2.0 Flash has a large context window (~1M tokens)
        self.max_tokens = 800000  # Conservative limit to account for conversation history
        self.encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
        self.max_retries = 1  # fail fast per model, rely on fallback chain instead
        self.retry_delay = 5  # seconds

    def count_tokens(self, text):
        """Count the number of tokens in a text"""
        return len(self.encoding.encode(text))

    def extract_repo_info(self, github_url):
        """Extract repository information using gitingest"""
        print(f"Extracting information from {github_url}...")
        try:
            # Use gitingest to extract repo information
            summary, tree, content = ingest(github_url)
            
            # Check token counts for each component
            summary_tokens = self.count_tokens(summary)
            tree_tokens = self.count_tokens(tree)
            content_tokens = self.count_tokens(content)
            
            print(f"Token counts - Summary: {summary_tokens}, Tree: {tree_tokens}, Content: {content_tokens}")
            
            # Calculate how much content we can include
            header = f"SUMMARY:\n{summary}\n\nFILE STRUCTURE:\n{tree}\n\nCONTENT:\n"
            header_tokens = self.count_tokens(header)
            
            # Reserve more space for conversation
            conversation_buffer = 100000  # Reserve 100K tokens for conversation
            max_content_tokens = self.max_tokens - header_tokens - conversation_buffer
            
            # Truncate content if needed
            if content_tokens > max_content_tokens:
                print(f"Warning: Content exceeds available token space. Truncating from {content_tokens} to {max_content_tokens} tokens.")
                content_token_list = self.encoding.encode(content)
                truncated_content = self.encoding.decode(content_token_list[:max_content_tokens])
                content = truncated_content
            
            # Combine all the information
            repo_info = f"SUMMARY:\n{summary}\n\nFILE STRUCTURE:\n{tree}\n\nCONTENT:\n{content}"
            
            # Final token count check
            token_count = self.count_tokens(repo_info)
            print(f"Repository information extracted. Token count: {token_count}")
            
            # Safety check
            if token_count > self.max_tokens:
                print(f"Warning: Repository information still exceeds the token limit. Performing additional truncation.")
                repo_info_tokens = self.encoding.encode(repo_info)
                repo_info = self.encoding.decode(repo_info_tokens[:self.max_tokens - conversation_buffer])
                token_count = self.count_tokens(repo_info)
                print(f"Final token count after truncation: {token_count}")
            
            self.repo_info = repo_info
            self.token_count = token_count
            return True
        except Exception as e:
            print(f"Error extracting repository information: {e}")
            return False

    def add_to_conversation(self, role, content):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def create_system_prompt(self):
        """Create the system prompt with repository information"""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Calculate tokens for the system prompt
        base_prompt = (
            f"You are GitHub Navigator, an AI assistant specialized in helping users with GitHub repositories. "
            f"Today is {current_date}. "
            f"You have been provided with information about a GitHub repository. "
            f"Use this information to help the user understand and work with this repository. "
            f"Be concise, accurate, and helpful. If asked questions about the repository content, "
            f"refer to the provided information to give accurate answers."
        )
        
        base_prompt_tokens = self.count_tokens(base_prompt)
        repo_info_tokens = self.count_tokens(self.repo_info)
        
        print(f"System prompt base tokens: {base_prompt_tokens}, Repo info tokens: {repo_info_tokens}")
        
        # Check if total tokens would be too large
        total_tokens = base_prompt_tokens + repo_info_tokens
        if total_tokens > 700000:  # Close to Gemini's practical limit
            print(f"Warning: System prompt would be too large ({total_tokens} tokens). Trimming repository information.")
            
            # Extract the important parts
            parts = self.repo_info.split("\n\n")
            if len(parts) >= 3:  # Should have SUMMARY, FILE STRUCTURE, and CONTENT
                summary = parts[0]
                file_structure = parts[1]
                
                # Calculate how much content we can include
                max_content_tokens = 650000 - self.count_tokens(base_prompt) - self.count_tokens(summary) - self.count_tokens(file_structure) - 100
                
                content_parts = self.repo_info.split("CONTENT:\n")
                if len(content_parts) > 1:
                    content = content_parts[1]
                    content_tokens = self.count_tokens(content)
                    
                    if content_tokens > max_content_tokens:
                        content_token_list = self.encoding.encode(content)
                        truncated_content = self.encoding.decode(content_token_list[:max_content_tokens])
                        trimmed_repo_info = f"{summary}\n\n{file_structure}\n\nCONTENT:\n{truncated_content}"
                    else:
                        trimmed_repo_info = self.repo_info
                else:
                    trimmed_repo_info = f"{summary}\n\n{file_structure}\n\nCONTENT: [Content too large to include]"
            else:
                # Just truncate if we can't parse the structure
                repo_info_tokens = self.encoding.encode(self.repo_info)
                max_tokens = 650000 - self.count_tokens(base_prompt) - 100
                trimmed_repo_info = self.encoding.decode(repo_info_tokens[:max_tokens])
            
            # Final check
            final_system_prompt = f"{base_prompt}\n\n{trimmed_repo_info}"
            print(f"Final system prompt tokens: {self.count_tokens(final_system_prompt)}")
            return final_system_prompt
        
        # If not too large, return the full system prompt
        return f"{base_prompt}\n\n{self.repo_info}"

    def chat(self, user_message):
        """Process user message and generate a response"""
        if not self.repo_info:
            # Check if this is a GitHub URL
            github_url_pattern = r'https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+'
            match = re.search(github_url_pattern, user_message)
            
            if match:
                github_url = match.group(0)
                success = self.extract_repo_info(github_url)
                if success:
                    self.add_to_conversation("system", self.create_system_prompt())
                    self.add_to_conversation("user", f"I want to work with the repository at {github_url}. Please help me understand it.")
                    return self.generate_response()
                else:
                    return "I had trouble extracting information from that repository. Please check the URL and try again."
            else:
                return "Please provide a valid GitHub repository URL to get started."
        
        # Add user message to conversation history
        self.add_to_conversation("user", user_message)
        
        # Generate response
        return self.generate_response()

    def _build_messages(self):
        """Build the messages array for the API call"""
        messages = []
        system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
        if system_messages:
            messages.append(system_messages[-1])
        for msg in self.conversation_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append(msg)
        return messages

    def _try_openrouter(self, messages):
        """Try OpenRouter models with fallback chain"""
        if not self.openrouter_client:
            return None

        for model in self.openrouter_models:
            logger.info(f"Trying OpenRouter model: {model}")
            for attempt in range(self.max_retries):
                try:
                    response = self.openrouter_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        extra_headers={
                            "HTTP-Referer": self.site_url,
                            "X-Title": self.site_name,
                        }
                    )
                    self.model = model
                    return response.choices[0].message.content
                except Exception as e:
                    err = str(e).lower()
                    if "404" in str(e) or "no endpoints" in err:
                        logger.warning(f"Model {model} not available, skipping")
                        break
                    if "429" in str(e) or "rate_limit" in err or "rate limit" in err or "quota" in err:
                        logger.warning(f"Rate limited on {model} (attempt {attempt + 1}/{self.max_retries})")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                        else:
                            break
                    else:
                        logger.error(f"OpenRouter error: {e}")
                        if "context" in err and "limit" in err:
                            return "The repository is too large to process. Please try a smaller repository."
                        return f"OpenRouter API error: {e}"
        return None

    def _try_google(self, messages):
        """Try Google Gemini models with fallback"""
        if not self.google_client:
            return None

        # Convert chat messages to Gemini format
        system_text = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        config = {}
        if system_text:
            config["system_instruction"] = system_text

        for model in self.google_models:
            logger.info(f"Trying Google model: {model}")
            for attempt in range(self.max_retries):
                try:
                    response = self.google_client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    self.model = f"google/{model}"
                    return response.text
                except Exception as e:
                    err = str(e).lower()
                    if "429" in str(e) or "rate" in err or "quota" in err:
                        logger.warning(f"Rate limited on Google {model} (attempt {attempt + 1}/{self.max_retries})")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                        else:
                            break
                    else:
                        logger.error(f"Google Gemini error: {e}")
                        return f"Google Gemini error: {e}"
        return None

    def generate_response(self):
        """Generate a response with full provider fallback: OpenRouter -> Google Gemini"""
        messages = self._build_messages()

        # Try Google Gemini first
        result = self._try_google(messages)
        if result:
            self.add_to_conversation("assistant", result)
            return result

        # Fall back to OpenRouter
        logger.info("Google models exhausted, falling back to OpenRouter")
        result = self._try_openrouter(messages)
        if result:
            self.add_to_conversation("assistant", result)
            return result

        return "All models are currently unavailable. Please wait a minute and try again."

    def save_conversation(self, filename="conversation.json"):
        """Save the current conversation to a file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            print(f"Conversation saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def load_conversation(self, filename="conversation.json"):
        """Load a conversation from a file"""
        try:
            with open(filename, 'r') as f:
                self.conversation_history = json.load(f)
            print(f"Conversation loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filename}.")
        except Exception as e:
            print(f"Error loading conversation: {e}")

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Navigator Chatbot")
    parser.add_argument("--api-key", help="OpenRouter API Key (or set OPENROUTER_API_KEY environment variable)")
    parser.add_argument("--load", help="Load conversation from file")
    args = parser.parse_args()
    
    try:
        # Check for API key in command line args first, then environment
        api_key = args.api_key
        if not api_key:
            # Get from environment with proper logging
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                logger.info(f"Using API key from environment: {api_key[:5]}...{api_key[-5:]}")
            else:
                print("Error: OpenRouter API key not configured. Please provide an API key.")
                print("Usage: python github_companion.py --api-key YOUR_API_KEY")
                print("       or set the OPENROUTER_API_KEY environment variable")
                sys.exit(1)
        
        # Initialize the companion with the API key
        companion = GitHubCompanion(openrouter_api_key=api_key)
        
        if args.load:
            companion.load_conversation(args.load)
        
        print("GitHub Companion Bot - Your AI assistant for GitHub repositories")
        print("Enter a GitHub repository URL to begin, or type 'exit' to quit")
        
        while True:
            try:
                user_input = input("\nYou: ")
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Saving conversation...")
                    companion.save_conversation()
                    print("Goodbye!")
                    break
                
                response = companion.chat(user_input)
                print(f"\nGitHub Companion: {response}")
            except KeyboardInterrupt:
                print("\nSaving conversation and exiting...")
                companion.save_conversation()
                print("Goodbye!")
                break
            except Exception as e:
                print(f"Error processing input: {e}")
    except Exception as e:
        logger.error(f"Error initializing GitHub Companion: {e}")
        print(f"Error initializing GitHub Companion: {e}")
        print("Please check your dependencies and API key configuration.")
        sys.exit(1) 