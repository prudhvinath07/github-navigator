"""
Requesty provider (existing implementation).
"""

import openai
from typing import List, Dict
import logging
from .base_provider import BaseProvider, ChatResponse, ProviderConfig

logger = logging.getLogger(__name__)

class RequestyProvider(BaseProvider):
    """Requesty provider implementation (compatible with existing code)."""
    
    def _initialize_client(self):
        """Initialize the Requesty client."""
        try:
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://router.requesty.ai/v1"
            )
            logger.info(f"Requesty client initialized for provider: {self.name}")
        except Exception as e:
            logger.error(f"Failed to initialize Requesty client: {e}")
            raise
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> ChatResponse:
        """Generate a chat completion using Requesty."""
        try:
            # Prepare the request
            request_params = {
                "model": model,
                "messages": messages,
                "timeout": self.config.timeout,
                **kwargs
            }
            
            # Make the API call
            response = self._client.chat.completions.create(**request_params)
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Extract usage information if available
            usage_info = None
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            
            return ChatResponse(
                content=content,
                model=model,
                provider=self.name,
                usage=usage_info
            )
            
        except openai.RateLimitError as e:
            logger.warning(f"Requesty rate limit exceeded: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"Requesty API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Requesty chat completion: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available models from Requesty."""
        # Return configured models since Requesty doesn't have a standard models endpoint
        return self.config.models or [
            "google/gemini-2.0-flash-thinking-exp-01-21",
            "google/gemini-2.5-pro-exp-03-25"
        ]