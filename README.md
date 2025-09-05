# JabberTTS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)

The fastest, most feature-rich, easy to use, and efficient Text-to-Speech API with near-instant response times even on CPU-only systems.

## 🚀 Quick Start

Get JabberTTS running in under 2 minutes:

```bash
# Clone the repository
git clone https://github.com/TaskWizer/JabberTTS.git
cd JabberTTS

# Install and run (uv handles dependencies automatically)
uv run python app.py
```

The API will be available at `http://localhost:8000` with OpenAI-compatible endpoints.

## ✨ Key Features

- **🏃‍♂️ Lightning Fast**: RTF < 0.5 on CPU-only hardware (4-core, 4GB RAM)
- **🔄 OpenAI Compatible**: Drop-in replacement for OpenAI TTS API
- **🎭 Voice Cloning**: Create custom voices from 10-30 second audio samples
- **💾 Resource Efficient**: < 2GB memory usage, optimized for constrained environments
- **🐳 Easy Deployment**: Single command setup with Docker support
- **🔊 High Quality**: MOS > 3.8, professional-grade audio output
- **🌐 Multiple Formats**: MP3, WAV, FLAC, Opus, AAC support
- **📡 Streaming**: Real-time audio streaming for long text

## 🎯 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Real-Time Factor (RTF) | < 0.5 | 🔄 In Development |
| Memory Usage | < 2GB | 🔄 In Development |
| First Token Latency | < 1s | 🔄 In Development |
| Audio Quality (MOS) | > 3.8 | 🔄 In Development |
| API Compatibility | 100% OpenAI | 🔄 In Development |

## 📖 Documentation

### 📋 Planning & Strategy
- **[📊 Project Plan](docs/PLAN.md)** - Development roadmap with weekly milestones and success criteria
- **[📝 Technical Specifications](docs/SPEC.md)** - Functional and technical requirements, API contracts
- **[🎯 Product Requirements](docs/PRD.md)** - User personas, use cases, and acceptance criteria

### 🔧 Technical Reference
- **[🏗️ Architecture Reference](docs/REFERENCE.md)** - Implementation guide, code examples, optimization strategies
- **[🔬 Research & Analysis](docs/RESEARCH.md)** - Technology comparisons and architectural decisions
- **[🧪 Testing Strategy](docs/TESTING.md)** - Comprehensive testing approach with 80% coverage requirement

### 📋 Project Management
- **[✅ Task Breakdown](docs/TASKS.md)** - Hierarchical task list with dependencies and estimates
- **[📅 Changelog](docs/CHANGELOG.md)** - Version history and release notes
- **[🤝 Contributing Guide](docs/CONTRIBUTING.md)** - Development guidelines and code standards

## 🛠️ Technology Stack

- **🚀 API Framework**: FastAPI with async support
- **🧠 Inference Engine**: ONNX Runtime (CPU-optimized)
- **🎵 Core Model**: OpenAudio S1-mini (optimized)
- **📝 Text Processing**: eSpeak-NG via phonemizer
- **🎧 Audio Encoding**: FFmpeg with multiple format support
- **📦 Package Management**: uv for dependency management

## 🔌 API Usage

### Basic Text-to-Speech

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openaudio-s1-mini",
    "input": "Hello, this is JabberTTS speaking!",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "openaudio-s1-mini",
        "input": "Welcome to JabberTTS!",
        "voice": "alloy",
        "speed": 1.0
    }
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### Voice Cloning

```bash
# Upload a voice sample
curl -X POST "http://localhost:8000/v1/voices" \
  -F "file=@voice_sample.wav" \
  -F "name=my_custom_voice"

# Use the custom voice
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openaudio-s1-mini",
    "input": "This is my cloned voice!",
    "voice": "my_custom_voice"
  }' \
  --output cloned_speech.mp3
```

## 🐳 Docker Deployment

```bash
# Build the container
docker build -t jabbertts .

# Run the service
docker run -p 8000:8000 jabbertts

# Or use docker-compose
docker-compose up -d
```

## ⚙️ Configuration

JabberTTS can be configured via environment variables:

```bash
# Server configuration
export JABBERTTS_HOST=0.0.0.0
export JABBERTTS_PORT=8000
export JABBERTTS_WORKERS=1

# Model configuration
export JABBERTTS_MODEL_PATH=/models/openaudio-s1-mini.onnx
export JABBERTTS_MAX_TEXT_LENGTH=4096

# Performance tuning
export JABBERTTS_MAX_CONCURRENT_REQUESTS=100
export JABBERTTS_REQUEST_TIMEOUT=30

# Authentication (optional)
export JABBERTTS_API_KEY=your-secret-key
export JABBERTTS_RATE_LIMIT=100  # requests per minute
```

## 🏗️ Development Setup

### Prerequisites
- Python 3.9+ (recommended: 3.11)
- uv package manager
- Git
- Docker (optional)

### Development Installation

```bash
# Clone the repository
git clone https://github.com/TaskWizer/JabberTTS.git
cd JabberTTS

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup development environment
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Start development server
uv run python app.py
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=jabbertts --cov-report=html

# Run performance tests
uv run pytest -m performance

# Run load tests
uv run locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on:

- Development workflow and branch strategy
- Code standards and quality requirements
- Testing guidelines and coverage requirements
- Pull request process and review criteria

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure tests pass: `uv run pytest`
5. Commit your changes: `git commit -m 'feat: add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📊 Project Status

### Current Phase: Foundation & Planning ✅
- [x] Project structure and documentation
- [x] Comprehensive planning and specifications
- [x] Testing strategy and framework
- [x] Development guidelines and standards

### Next Phase: Core Development 🔄
- [ ] FastAPI server implementation
- [ ] OpenAudio S1-mini model integration
- [ ] Basic inference engine
- [ ] OpenAI-compatible API endpoints

### Roadmap
- **v0.2.0** (Week 4): MVP with basic TTS functionality
- **v0.3.0** (Week 7): Voice cloning capabilities
- **v0.4.0** (Week 9): Performance optimization (RTF < 0.5)
- **v1.0.0** (Week 12): Production-ready release

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAudio Team** for the S1-mini model
- **ONNX Runtime** for high-performance inference
- **FastAPI** for the excellent web framework
- **Coqui TTS** for TTS research and inspiration
- **Community Contributors** for feedback and improvements

## 📞 Support

- **📋 Issues**: [GitHub Issues](https://github.com/TaskWizer/JabberTTS/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/TaskWizer/JabberTTS/discussions)
- **📖 Documentation**: [Project Docs](docs/)
- **🔧 Contributing**: [Contributing Guide](docs/CONTRIBUTING.md)

---

**Made with ❤️ by the JabberTTS Team**

*Democratizing high-quality text-to-speech technology for everyone.*
