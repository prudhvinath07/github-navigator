"""
Configuration loader for provider settings.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
import re
import logging
from providers.base_provider import ProviderConfig

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Loads and manages provider configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Look for config file in current directory first, then in config directory
        current_dir = os.getcwd()
        config_files = [
            os.path.join(current_dir, "providers.yaml"),
            os.path.join(current_dir, "config", "providers.yaml"),
            os.path.join(current_dir, "config", "providers.json"),
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                return config_file
        
        # Default to yaml in config directory
        return os.path.join(current_dir, "config", "providers.yaml")
    
    def load_config(self, reload: bool = False) -> Dict[str, Any]:
        """Load configuration from file."""
        if self._config_cache and not reload:
            return self._config_cache
        
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
            
            # Expand environment variables
            config = self._expand_env_vars(config)
            self._config_cache = config
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def get_provider_configs(self) -> Dict[str, ProviderConfig]:
        """Get all provider configurations."""
        config = self.load_config()
        providers_config = config.get('providers', {})
        
        provider_configs = {}
        
        for name, provider_data in providers_config.items():
            try:
                # Ensure name is set
                provider_data['name'] = provider_data.get('name', name)
                
                # Create ProviderConfig
                provider_config = ProviderConfig(**provider_data)
                
                # Only include if enabled and has API key
                if provider_config.enabled and provider_config.api_key:
                    provider_configs[name] = provider_config
                else:
                    logger.info(f"Skipping provider {name}: disabled or missing API key")
                    
            except Exception as e:
                logger.warning(f"Failed to create config for provider {name}: {e}")
                continue
        
        return provider_configs
    
    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        configs = self.get_provider_configs()
        return configs.get(name)
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(obj, dict):
            return {key: self._expand_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._expand_string_env_vars(obj)
        else:
            return obj
    
    def _expand_string_env_vars(self, text: str) -> str:
        """Expand environment variables in a string."""
        # Pattern matches ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            else:
                return default_value
        
        return re.sub(pattern, replace_var, text)
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate all provider configurations."""
        configs = self.get_provider_configs()
        validation_results = {}
        
        for name, config in configs.items():
            try:
                # Basic validation
                is_valid = True
                issues = []
                
                if not config.api_key or config.api_key.startswith("${"):
                    is_valid = False
                    issues.append("Missing or unexpanded API key")
                
                if not config.models:
                    issues.append("No models specified")
                
                if issues:
                    logger.warning(f"Provider {name} validation issues: {', '.join(issues)}")
                
                validation_results[name] = is_valid
                
            except Exception as e:
                logger.error(f"Error validating provider {name}: {e}")
                validation_results[name] = False
        
        return validation_results