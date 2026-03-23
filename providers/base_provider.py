"""
Base provider class for AI API integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    name: str
    api_key: str
    base_url: Optional[str] = None
    priority: int = 1
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    models: List[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.models is None:
            self.models = []

@dataclass
class ChatResponse:
    """Response from a chat completion."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    response_time: float = 0.0

class BaseProvider(ABC):
    """Abstract base class for AI providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self._client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider's client."""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> ChatResponse:
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        if not self.config.api_key:
            return False
        if not self.config.name:
            return False
        return True
    
    def is_enabled(self) -> bool:
        """Check if provider is enabled."""
        return self.config.enabled and self.validate_config()
    
    def supports_model(self, model: str) -> bool:
        """Check if provider supports a specific model."""
        if not self.config.models:
            return True  # If no models specified, assume all supported
        return model in self.config.models or any(model.startswith(m) for m in self.config.models)
    
    def get_priority(self) -> int:
        """Get provider priority (lower number = higher priority)."""
        return self.config.priority
    
    def chat_with_retry(self, messages: List[Dict[str, str]], model: str, **kwargs) -> ChatResponse:
        """Chat completion with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = self.chat_completion(messages, model, **kwargs)
                response.response_time = time.time() - start_time
                return response
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                break
        
        raise last_exception or Exception(f"Failed to get response from {self.name} after {self.config.max_retries} attempts")