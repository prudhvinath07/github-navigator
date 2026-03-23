"""
OpenRouter provider for accessing multiple models through a single API.
"""

import openai
from typing import List, Dict
import logging
from .base_provider import BaseProvider, ChatResponse, ProviderConfig

logger = logging.getLogger(__name__)

class OpenRouterProvider(BaseProvider):
    """OpenRouter provider implementation."""
    
    def _initialize_client(self):
        """Initialize the OpenRouter client."""
        try:
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://openrouter.ai/api/v1"
            )
            logger.info(f"OpenRouter client initialized for provider: {self.name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> ChatResponse:
        """Generate a chat completion using OpenRouter."""
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
            logger.warning(f"OpenRouter rate limit exceeded: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenRouter chat completion: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available models from OpenRouter."""
        try:
            # OpenRouter provides a models endpoint
            models_response = self._client.get("/models")
            if hasattr(models_response, 'data'):
                return [model.id for model in models_response.data]
            else:
                # Fallback to configured models
                return self.config.models or []
        except Exception as e:
            logger.warning(f"Failed to fetch available models from OpenRouter: {e}")
            # Return configured models as fallback
            return self.config.models or []
    
    def validate_config(self) -> bool:
        """Validate OpenRouter configuration."""
        if not super().validate_config():
            return False
        
        # Check if base_url is set (required for OpenRouter)
        if not self.config.base_url:
            logger.error("OpenRouter requires base_url to be configured")
            return False
        
        return True