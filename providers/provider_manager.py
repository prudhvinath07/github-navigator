"""
Provider manager for handling multiple AI providers.
"""

from typing import Dict, List, Optional, Any
import logging
from .base_provider import BaseProvider, ChatResponse, ProviderConfig

logger = logging.getLogger(__name__)

class ProviderManager:
    """Manages multiple AI providers with fallback and routing logic."""
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._fallback_order: List[str] = []
    
    def register_provider(self, provider: BaseProvider):
        """Register a new provider."""
        if not provider.is_enabled():
            logger.warning(f"Provider {provider.name} is disabled or has invalid config")
            return
        
        self._providers[provider.name] = provider
        self._update_fallback_order()
        logger.info(f"Registered provider: {provider.name}")
    
    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Get a specific provider by name."""
        return self._providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())
    
    def list_enabled_providers(self) -> List[str]:
        """List enabled provider names."""
        return [name for name, provider in self._providers.items() if provider.is_enabled()]
    
    def get_providers_for_model(self, model: str) -> List[BaseProvider]:
        """Get all providers that support a specific model."""
        providers = []
        for provider in self._providers.values():
            if provider.is_enabled() and provider.supports_model(model):
                providers.append(provider)
        
        # Sort by priority (lower number = higher priority)
        providers.sort(key=lambda p: p.get_priority())
        return providers
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = None, 
                       provider_name: str = None,
                       fallback: bool = True,
                       **kwargs) -> ChatResponse:
        """
        Generate chat completion with provider selection and fallback.
        
        Args:
            messages: List of chat messages
            model: Specific model to use
            provider_name: Specific provider to use
            fallback: Whether to try other providers on failure
            **kwargs: Additional arguments for the provider
        """
        
        # If specific provider requested, try that first
        if provider_name:
            provider = self.get_provider(provider_name)
            if not provider:
                raise ValueError(f"Provider '{provider_name}' not found")
            
            if not provider.is_enabled():
                raise ValueError(f"Provider '{provider_name}' is disabled")
            
            if model and not provider.supports_model(model):
                raise ValueError(f"Provider '{provider_name}' does not support model '{model}'")
            
            try:
                return provider.chat_with_retry(messages, model or self._get_default_model(provider), **kwargs)
            except Exception as e:
                if not fallback:
                    raise
                logger.warning(f"Provider {provider_name} failed: {e}. Trying fallback providers...")
        
        # Try providers based on model support and priority
        providers_to_try = []
        
        if model:
            providers_to_try = self.get_providers_for_model(model)
        else:
            # Use fallback order if no specific model
            providers_to_try = [self._providers[name] for name in self._fallback_order if name in self._providers]
        
        if not providers_to_try:
            raise ValueError("No available providers found")
        
        last_exception = None
        
        for provider in providers_to_try:
            # Skip if we already tried this provider
            if provider_name and provider.name == provider_name:
                continue
                
            try:
                effective_model = model or self._get_default_model(provider)
                logger.info(f"Trying provider: {provider.name} with model: {effective_model}")
                return provider.chat_with_retry(messages, effective_model, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Provider {provider.name} failed: {e}")
                continue
        
        raise last_exception or Exception("All providers failed")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models for each provider."""
        models = {}
        for name, provider in self._providers.items():
            if provider.is_enabled():
                try:
                    models[name] = provider.get_available_models()
                except Exception as e:
                    logger.warning(f"Failed to get models from {name}: {e}")
                    models[name] = []
        return models
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health = {}
        for name, provider in self._providers.items():
            try:
                # Simple test message
                test_messages = [{"role": "user", "content": "Hello"}]
                provider.chat_completion(test_messages, self._get_default_model(provider))
                health[name] = True
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                health[name] = False
        return health
    
    def _update_fallback_order(self):
        """Update the fallback order based on provider priorities."""
        enabled_providers = [(name, provider) for name, provider in self._providers.items() 
                           if provider.is_enabled()]
        
        # Sort by priority (lower number = higher priority)
        enabled_providers.sort(key=lambda x: x[1].get_priority())
        
        self._fallback_order = [name for name, _ in enabled_providers]
        logger.info(f"Updated fallback order: {self._fallback_order}")
    
    def _get_default_model(self, provider: BaseProvider) -> str:
        """Get a default model for a provider."""
        if provider.config.models:
            return provider.config.models[0]
        return "default"  # Fallback model name