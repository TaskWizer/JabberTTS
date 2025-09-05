"""JabberTTS Configuration Management.

This module handles all configuration settings for JabberTTS, including
environment variable loading, validation, and default values.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """JabberTTS application settings.
    
    All settings can be configured via environment variables with the
    JABBERTTS_ prefix (e.g., JABBERTTS_HOST=0.0.0.0).
    """
    
    # Server Configuration
    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    
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
    model_name: str = Field(default="openaudio-s1-mini", description="Model identifier")
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


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings.
    
    Returns:
        Application settings instance
    """
    return Settings()


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
