"""Basic tests to verify JabberTTS setup and functionality."""

import pytest
from fastapi.testclient import TestClient

from jabbertts.main import create_app
from jabbertts import __version__


class TestBasicFunctionality:
    """Test basic application functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_version_import(self):
        """Test that version can be imported."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__.split(".")) >= 2
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__
        assert data["service"] == "jabbertts"
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "JabberTTS"
        assert data["version"] == __version__
        assert "description" in data
    
    def test_voices_endpoint(self, client):
        """Test voices listing endpoint."""
        response = client.get("/v1/voices")
        assert response.status_code == 200
        
        data = response.json()
        assert "voices" in data
        assert isinstance(data["voices"], list)
        assert len(data["voices"]) > 0
        
        # Check that default voices are present
        voice_ids = [voice["id"] for voice in data["voices"]]
        expected_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        for expected_voice in expected_voices:
            assert expected_voice in voice_ids
    
    def test_speech_endpoint_validation(self, client):
        """Test speech endpoint input validation."""
        # Test empty input
        response = client.post("/v1/audio/speech", json={
            "model": "openaudio-s1-mini",
            "input": "",
            "voice": "alloy"
        })
        assert response.status_code == 422  # Pydantic validation error
        
        # Test too long input
        long_text = "a" * 5000
        response = client.post("/v1/audio/speech", json={
            "model": "openaudio-s1-mini",
            "input": long_text,
            "voice": "alloy"
        })
        assert response.status_code == 422  # Pydantic validation error
    
    def test_speech_endpoint_basic(self, client):
        """Test basic speech generation endpoint."""
        response = client.post("/v1/audio/speech", json={
            "model": "openaudio-s1-mini",
            "input": "Hello, world!",
            "voice": "alloy",
            "response_format": "mp3"
        })
        
        # Should return 200 even with placeholder implementation
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0
        
        # Check that the default model is present
        model_ids = [model["id"] for model in data["models"]]
        assert "openaudio-s1-mini" in model_ids


class TestAPIValidation:
    """Test API request validation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_invalid_voice(self, client):
        """Test invalid voice parameter."""
        response = client.post("/v1/audio/speech", json={
            "model": "openaudio-s1-mini",
            "input": "Hello, world!",
            "voice": "invalid@voice!",
            "response_format": "mp3"
        })
        assert response.status_code == 422
    
    def test_invalid_speed(self, client):
        """Test invalid speed parameter."""
        response = client.post("/v1/audio/speech", json={
            "model": "openaudio-s1-mini",
            "input": "Hello, world!",
            "voice": "alloy",
            "speed": 10.0  # Too high
        })
        assert response.status_code == 422
    
    def test_invalid_format(self, client):
        """Test invalid response format."""
        response = client.post("/v1/audio/speech", json={
            "model": "openaudio-s1-mini",
            "input": "Hello, world!",
            "voice": "alloy",
            "response_format": "invalid"
        })
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test missing required fields."""
        response = client.post("/v1/audio/speech", json={
            "model": "openaudio-s1-mini",
            # Missing 'input' field
            "voice": "alloy"
        })
        assert response.status_code == 422


class TestConfiguration:
    """Test configuration management."""
    
    def test_settings_import(self):
        """Test that settings can be imported and created."""
        from jabbertts.config import get_settings
        
        settings = get_settings()
        assert settings is not None
        assert settings.host is not None
        assert settings.port > 0
        assert settings.model_name is not None
    
    def test_environment_info(self):
        """Test environment info function."""
        from jabbertts.config import get_environment_info
        
        env_info = get_environment_info()
        assert "python_version" in env_info
        assert "platform" in env_info
        assert "cwd" in env_info
        assert "env_vars" in env_info
