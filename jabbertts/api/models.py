"""JabberTTS API Models.

This module defines Pydantic models for request/response validation
and OpenAPI documentation generation.
"""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class VoiceEnum(str, Enum):
    """Available voice options."""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class ResponseFormatEnum(str, Enum):
    """Available audio response formats."""
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class ModelEnum(str, Enum):
    """Available TTS models."""
    OPENAUDIO_S1_MINI = "openaudio-s1-mini"
    SPEECHT5 = "speecht5"


class TTSRequest(BaseModel):
    """Text-to-Speech request model.
    
    This model is fully compatible with OpenAI's TTS API request format.
    """
    
    model: ModelEnum = Field(
        default=ModelEnum.SPEECHT5,
        description="The TTS model to use for generation"
    )
    
    input: str = Field(
        ...,
        description="The text to convert to speech",
        min_length=1,
        max_length=4096,
        json_schema_extra={"example": "Hello, this is a test of the JabberTTS system."}
    )
    
    voice: Union[VoiceEnum, str] = Field(
        default=VoiceEnum.ALLOY,
        description="The voice to use for speech generation. Can be a built-in voice or custom voice ID.",
        json_schema_extra={"example": "alloy"}
    )
    
    response_format: ResponseFormatEnum = Field(
        default=ResponseFormatEnum.MP3,
        description="The audio format for the response"
    )
    
    speed: float = Field(
        default=1.0,
        description="The speed of speech generation",
        ge=0.25,
        le=4.0,
        json_schema_extra={"example": 1.0}
    )
    
    @field_validator("input")
    @classmethod
    def validate_input(cls, v):
        """Validate input text."""
        if not v.strip():
            raise ValueError("Input text cannot be empty or only whitespace")
        return v.strip()

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, v):
        """Validate voice parameter."""
        # Allow built-in voices and custom voice IDs
        if isinstance(v, str):
            # Custom voice IDs should be alphanumeric with hyphens/underscores
            if not v.replace("-", "").replace("_", "").isalnum():
                raise ValueError("Custom voice ID must be alphanumeric with optional hyphens or underscores")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "speecht5",
                "input": "Hello, this is JabberTTS speaking! How can I help you today?",
                "voice": "alloy",
                "response_format": "mp3",
                "speed": 1.0
            }
        }
    }


class TTSResponse(BaseModel):
    """Text-to-Speech response model."""
    
    audio_data: bytes = Field(
        ...,
        description="The generated audio data"
    )
    
    content_type: str = Field(
        ...,
        description="MIME type of the audio data"
    )
    
    duration: Optional[float] = Field(
        None,
        description="Duration of the generated audio in seconds"
    )


class Voice(BaseModel):
    """Voice information model."""
    
    id: str = Field(
        ...,
        description="Unique voice identifier"
    )
    
    name: str = Field(
        ...,
        description="Human-readable voice name"
    )
    
    description: Optional[str] = Field(
        None,
        description="Voice description"
    )
    
    type: str = Field(
        ...,
        description="Voice type (built-in, custom, cloned)"
    )
    
    language: Optional[str] = Field(
        None,
        description="Primary language code (e.g., 'en', 'es')"
    )
    
    created_at: Optional[str] = Field(
        None,
        description="Creation timestamp for custom voices"
    )


class VoiceListResponse(BaseModel):
    """Response model for voice listing."""
    
    voices: List[Voice] = Field(
        ...,
        description="List of available voices"
    )


class VoiceUploadRequest(BaseModel):
    """Voice upload request model."""
    
    name: str = Field(
        ...,
        description="Name for the custom voice",
        min_length=1,
        max_length=100
    )
    
    description: Optional[str] = Field(
        None,
        description="Optional description for the voice",
        max_length=500
    )
    
    language: Optional[str] = Field(
        None,
        description="Primary language of the voice sample",
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$"
    )


class VoiceUploadResponse(BaseModel):
    """Voice upload response model."""
    
    voice_id: str = Field(
        ...,
        description="Generated ID for the uploaded voice"
    )
    
    name: str = Field(
        ...,
        description="Name of the uploaded voice"
    )
    
    status: str = Field(
        ...,
        description="Processing status"
    )
    
    message: str = Field(
        ...,
        description="Status message"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: "ErrorDetail" = Field(
        ...,
        description="Error details"
    )


class ErrorDetail(BaseModel):
    """Error detail model."""
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    type: str = Field(
        ...,
        description="Error type identifier"
    )
    
    param: Optional[str] = Field(
        None,
        description="Parameter that caused the error"
    )
    
    code: Optional[str] = Field(
        None,
        description="Error code"
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(
        ...,
        description="Service health status"
    )
    
    version: str = Field(
        ...,
        description="Service version"
    )
    
    service: str = Field(
        ...,
        description="Service name"
    )
    
    timestamp: Optional[str] = Field(
        None,
        description="Response timestamp"
    )
    
    uptime: Optional[float] = Field(
        None,
        description="Service uptime in seconds"
    )


class MetricsResponse(BaseModel):
    """Metrics response model."""
    
    requests_total: int = Field(
        ...,
        description="Total number of requests processed"
    )
    
    requests_per_minute: float = Field(
        ...,
        description="Current requests per minute"
    )
    
    average_response_time: float = Field(
        ...,
        description="Average response time in seconds"
    )
    
    active_connections: int = Field(
        ...,
        description="Number of active connections"
    )
    
    memory_usage: float = Field(
        ...,
        description="Current memory usage in MB"
    )
    
    cpu_usage: float = Field(
        ...,
        description="Current CPU usage percentage"
    )


# Update forward references
ErrorResponse.model_rebuild()
