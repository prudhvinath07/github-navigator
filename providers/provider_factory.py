"""
Factory for creating provider instances.
"""

import logging
from typing import Dict, Type, Optional
from .base_provider import BaseProvider, ProviderConfig

logger = logging.getLogger(__name__)

class ProviderFactory:
    """Factory for creating provider instances."""
    
    _provider_classes: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register_provider_class(cls, name: str, provider_class: Type[BaseProvider]):
        """Register a provider class."""
        cls._provider_classes[name] = provider_class
        logger.info(f"Registered provider class: {name}")
    
    @classmethod
    def create_provider(cls, config: ProviderConfig) -> Optional[BaseProvider]:
        """Create a provider instance from configuration."""
        provider_class = cls._provider_classes.get(config.name)
        
        if not provider_class:
            logger.error(f"Unknown provider type: {config.name}")
            return None
        
        try:
            provider = provider_class(config)
            logger.info(f"Created provider instance: {config.name}")
            return provider
        except Exception as e:
            logger.error(f"Failed to create provider {config.name}: {e}")
            return None
    
    @classmethod
    def get_available_provider_types(cls) -> list[str]:
        """Get list of available provider types."""
        return list(cls._provider_classes.keys())
    
    @classmethod
    def is_provider_type_available(cls, name: str) -> bool:
        """Check if a provider type is available."""
        return name in cls._provider_classes

# Auto-register providers when they are imported
def auto_register_providers():
    """Automatically register available provider classes."""
    try:
        from .openrouter_provider import OpenRouterProvider
        ProviderFactory.register_provider_class("openrouter", OpenRouterProvider)
    except ImportError as e:
        logger.debug(f"OpenRouter provider not available: {e}")
    
    try:
        from .google_provider import GoogleProvider
        ProviderFactory.register_provider_class("google", GoogleProvider)
    except ImportError as e:
        logger.debug(f"Google provider not available: {e}")
    
    try:
        from .groq_provider import GroqProvider
        ProviderFactory.register_provider_class("groq", GroqProvider)
    except ImportError as e:
        logger.debug(f"Groq provider not available: {e}")
    
    try:
        from .anthropic_provider import AnthropicProvider
        ProviderFactory.register_provider_class("anthropic", AnthropicProvider)
    except ImportError as e:
        logger.debug(f"Anthropic provider not available: {e}")
    
    try:
        from .openai_provider import OpenAIProvider
        ProviderFactory.register_provider_class("openai", OpenAIProvider)
    except ImportError as e:
        logger.debug(f"OpenAI provider not available: {e}")
    
    try:
        from .together_provider import TogetherProvider
        ProviderFactory.register_provider_class("together", TogetherProvider)
    except ImportError as e:
        logger.debug(f"Together provider not available: {e}")
    
    try:
        from .requesty_provider import RequestyProvider
        ProviderFactory.register_provider_class("requesty", RequestyProvider)
    except ImportError as e:
        logger.debug(f"Requesty provider not available: {e}")

# Register providers on import
auto_register_providers()