"""JabberTTS Configuration Management.

This module handles all configuration settings for JabberTTS, including
JSON file loading, environment variable loading, validation, and default values.
Supports configuration precedence: CLI args > override.json > settings.json > env vars > defaults
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """JabberTTS application settings.
    
    All settings can be configured via environment variables with the
    JABBERTTS_ prefix (e.g., JABBERTTS_HOST=0.0.0.0).
    """
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host address (0.0.0.0 for network access, 127.0.0.1 for localhost only)")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    auto_port: bool = Field(default=False, description="Automatically find available port if configured port is in use")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    access_log: bool = Field(default=True, description="Enable access logging")
    
    # API Configuration
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    rate_limit: int = Field(default=100, description="Rate limit per minute")
    
    # Model Configuration
    model_path: Optional[str] = Field(default=None, description="Path to ONNX model file")
    model_name: str = Field(default="speecht5", description="Model identifier")
    max_text_length: int = Field(default=4096, description="Maximum input text length")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(default=100, description="Maximum concurrent requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    inference_timeout: int = Field(default=20, description="Model inference timeout")
    
    # Audio Configuration
    default_voice: str = Field(default="alloy", description="Default voice identifier")
    default_format: str = Field(default="mp3", description="Default audio format")
    default_speed: float = Field(default=1.0, description="Default speech speed")
    sample_rate: int = Field(default=24000, description="Audio sample rate")

    # Audio Quality Configuration
    audio_quality: str = Field(default="standard", description="Audio quality preset (low, standard, high, ultra)")
    enable_audio_enhancement: bool = Field(default=True, description="Enable audio post-processing enhancements")
    noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    dynamic_range_compression: bool = Field(default=True, description="Enable dynamic range compression")
    audio_normalization: str = Field(default="peak", description="Audio normalization method (peak, rms, lufs)")
    bitrate_quality: str = Field(default="adaptive", description="Bitrate quality (low, medium, high, adaptive)")
    stereo_enhancement: bool = Field(default=False, description="Enable stereo enhancement for mono sources")
    
    # Voice Cloning Configuration
    enable_voice_cloning: bool = Field(default=True, description="Enable voice cloning features")
    max_voice_sample_size: int = Field(default=50 * 1024 * 1024, description="Max voice sample size (50MB)")
    voice_storage_path: str = Field(default="./voices", description="Voice storage directory")
    
    # Cache Configuration
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Development Configuration
    debug: bool = Field(default=False, description="Enable debug mode")
    profiling: bool = Field(default=False, description="Enable performance profiling")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v):
        """Validate worker count."""
        if v < 1:
            raise ValueError("Workers must be at least 1")
        return v

    @field_validator("max_text_length")
    @classmethod
    def validate_max_text_length(cls, v):
        """Validate maximum text length."""
        if v < 1 or v > 100000:
            raise ValueError("Max text length must be between 1 and 100000")
        return v

    @field_validator("default_speed")
    @classmethod
    def validate_default_speed(cls, v):
        """Validate default speech speed."""
        if not 0.25 <= v <= 4.0:
            raise ValueError("Default speed must be between 0.25 and 4.0")
        return v

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v):
        """Validate audio sample rate."""
        valid_rates = [8000, 16000, 22050, 24000, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of: {valid_rates}")
        return v

    @field_validator("voice_storage_path")
    @classmethod
    def validate_voice_storage_path(cls, v):
        """Validate and create voice storage path."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

    @field_validator("audio_quality")
    @classmethod
    def validate_audio_quality(cls, v):
        """Validate audio quality preset."""
        valid_qualities = ["low", "standard", "high", "ultra"]
        if v.lower() not in valid_qualities:
            raise ValueError(f"Audio quality must be one of: {valid_qualities}")
        return v.lower()

    @field_validator("audio_normalization")
    @classmethod
    def validate_audio_normalization(cls, v):
        """Validate audio normalization method."""
        valid_methods = ["peak", "rms", "lufs"]
        if v.lower() not in valid_methods:
            raise ValueError(f"Audio normalization must be one of: {valid_methods}")
        return v.lower()

    @field_validator("bitrate_quality")
    @classmethod
    def validate_bitrate_quality(cls, v):
        """Validate bitrate quality setting."""
        valid_qualities = ["low", "medium", "high", "adaptive"]
        if v.lower() not in valid_qualities:
            raise ValueError(f"Bitrate quality must be one of: {valid_qualities}")
        return v.lower()
    
    def get_model_path(self) -> Path:
        """Get the model path, with fallback to default location."""
        if self.model_path:
            return Path(self.model_path)
        
        # Default model locations to check
        default_paths = [
            Path("./models/openaudio-s1-mini.onnx"),
            Path("./models/model.onnx"),
            Path(f"./{self.model_name}.onnx"),
        ]
        
        for path in default_paths:
            if path.exists():
                return path
        
        # Return the first default path for error messages
        return default_paths[0]
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug or self.reload or self.log_level == "DEBUG"
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins, handling development mode."""
        if self.is_development():
            return ["*"]
        return self.cors_origins
    
    model_config = {
        "env_prefix": "JABBERTTS_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


def load_json_config(config_dir: Union[str, Path] = "./config") -> Dict[str, Any]:
    """Load configuration from JSON files with proper precedence.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_dir)
    merged_config = {}

    # Load base settings.json
    settings_file = config_path / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                base_config = json.load(f)
                merged_config = _flatten_config(base_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {settings_file}: {e}")

    # Load override.json if it exists
    override_file = config_path / "override.json"
    if override_file.exists():
        try:
            with open(override_file, 'r', encoding='utf-8') as f:
                override_config = json.load(f)
                # Remove comment fields
                flattened_override = _flatten_config(override_config)
                merged_config.update(flattened_override)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {override_file}: {e}")

    return merged_config


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested configuration dictionary for Pydantic Settings.

    Maps JSON structure to Pydantic field names.

    Args:
        config: Nested configuration dictionary

    Returns:
        Flattened configuration dictionary with correct field names
    """
    flattened = {}

    # Mapping from JSON structure to Pydantic field names
    field_mapping = {
        'server': {
            'host': 'host',
            'port': 'port',
            'workers': 'workers',
            'reload': 'reload',
            'auto_port': 'auto_port'
        },
        'logging': {
            'log_level': 'log_level',
            'access_log': 'access_log'
        },
        'api': {
            'enable_docs': 'enable_docs',
            'cors_origins': 'cors_origins',
            'api_key': 'api_key',
            'rate_limit': 'rate_limit'
        },
        'model': {
            'model_path': 'model_path',
            'model_name': 'model_name',
            'max_text_length': 'max_text_length'
        },
        'performance': {
            'max_concurrent_requests': 'max_concurrent_requests',
            'request_timeout': 'request_timeout',
            'inference_timeout': 'inference_timeout'
        },
        'audio': {
            'default_voice': 'default_voice',
            'default_format': 'default_format',
            'default_speed': 'default_speed',
            'sample_rate': 'sample_rate',
            'audio_quality': 'audio_quality',
            'enable_audio_enhancement': 'enable_audio_enhancement',
            'noise_reduction': 'noise_reduction',
            'dynamic_range_compression': 'dynamic_range_compression',
            'audio_normalization': 'audio_normalization',
            'bitrate_quality': 'bitrate_quality',
            'stereo_enhancement': 'stereo_enhancement'
        },
        'voice_cloning': {
            'enable_voice_cloning': 'enable_voice_cloning',
            'max_voice_sample_size': 'max_voice_sample_size',
            'voice_storage_path': 'voice_storage_path'
        },
        'cache': {
            'enable_caching': 'enable_caching',
            'cache_ttl': 'cache_ttl'
        },
        'development': {
            'debug': 'debug',
            'profiling': 'profiling'
        }
    }

    for section_key, section_value in config.items():
        if section_key.startswith('_') or not isinstance(section_value, dict):
            continue

        if section_key in field_mapping:
            section_mapping = field_mapping[section_key]
            for field_key, field_value in section_value.items():
                if field_key.startswith('_'):
                    continue
                if field_key in section_mapping:
                    pydantic_field = section_mapping[field_key]
                    flattened[pydantic_field] = field_value

    return flattened


class ConfigurableSettings(Settings):
    """Enhanced Settings class with JSON configuration support."""

    def __init__(self, config_dir: Union[str, Path] = "./config", **kwargs):
        """Initialize settings with JSON configuration support.

        Args:
            config_dir: Directory containing configuration files
            **kwargs: Additional keyword arguments
        """
        # Load JSON configuration
        json_config = load_json_config(config_dir)

        # Merge with kwargs (CLI arguments take precedence)
        merged_kwargs = {**json_config, **kwargs}

        super().__init__(**merged_kwargs)


# Global settings instance for caching
_settings_instance: Optional[Settings] = None
_config_dir: Optional[str] = None


def get_settings(config_dir: Union[str, Path] = "./config", **cli_overrides) -> Settings:
    """Get application settings with JSON configuration support.

    Args:
        config_dir: Directory containing configuration files
        **cli_overrides: CLI argument overrides

    Returns:
        Application settings instance
    """
    global _settings_instance, _config_dir

    # Check if we need to reload settings
    config_dir_str = str(config_dir)
    if _settings_instance is None or _config_dir != config_dir_str or cli_overrides:
        _config_dir = config_dir_str
        _settings_instance = ConfigurableSettings(config_dir=config_dir, **cli_overrides)

    return _settings_instance


def reload_settings(config_dir: Union[str, Path] = "./config", **cli_overrides) -> Settings:
    """Force reload of application settings.

    Args:
        config_dir: Directory containing configuration files
        **cli_overrides: CLI argument overrides

    Returns:
        Reloaded application settings instance
    """
    global _settings_instance
    _settings_instance = None
    return get_settings(config_dir=config_dir, **cli_overrides)


def get_environment_info() -> dict:
    """Get environment information for debugging.
    
    Returns:
        Dictionary with environment information
    """
    return {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "platform": os.sys.platform,
        "cwd": str(Path.cwd()),
        "env_vars": {
            key: value for key, value in os.environ.items() 
            if key.startswith("JABBERTTS_")
        }
    }
