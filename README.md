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

## 🚀 Advanced Usage

### Command-Line Options

JabberTTS supports comprehensive command-line configuration:

```bash
# Basic usage
python app.py

# Custom port and host
python app.py --port 8001 --host 127.0.0.1

# High-quality audio with debug mode
python app.py --audio-quality high --debug

# Custom configuration directory
python app.py --config ./my-config --log-level DEBUG

# Development mode with auto-reload
python app.py --reload --debug

# Show all available options
python app.py --help

# Show version
python app.py --version
```

### Configuration System

JabberTTS uses a flexible configuration system with the following precedence (highest to lowest):

1. **Command-line arguments** (highest priority)
2. **override.json** file
3. **settings.json** file
4. **Environment variables** (JABBERTTS_*)
5. **Default values** (lowest priority)

#### JSON Configuration Files

Create configuration files in the `./config/` directory:

**config/settings.json** (base configuration):
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": false
  },
  "audio": {
    "audio_quality": "standard",
    "enable_audio_enhancement": true,
    "noise_reduction": true,
    "stereo_enhancement": false
  },
  "development": {
    "debug": false,
    "profiling": false
  }
}
```

**config/override.json** (your customizations):
```json
{
  "server": {
    "port": 8001,
    "auto_port": true
  },
  "audio": {
    "audio_quality": "high",
    "stereo_enhancement": true
  },
  "development": {
    "debug": true
  }
}
```

Copy `config/override.example.json` to `config/override.json` and customize as needed.

### Dashboard with Audio Playback

Access the interactive dashboard at `http://localhost:8000/dashboard/` featuring:

- **🎵 Real-time TTS Generation** - Generate speech with live preview
- **🔊 Automatic Audio Playback** - Generated audio plays automatically
- **🎛️ Audio Controls** - Play, pause, stop, and volume controls
- **💾 Download Support** - Download generated audio in various formats
- **📊 Live Metrics** - Real-time performance monitoring
- **🎚️ Quality Settings** - Adjust audio quality and enhancement options

### Real-Time Metrics

Monitor system performance with live metrics:

- **RTF (Real-Time Factor)** - Generation speed vs. audio duration
- **Response Time** - Average API response time
- **System Resources** - CPU and memory usage
- **Request Statistics** - Success rate and error tracking
- **Uptime Monitoring** - System availability tracking

Access metrics via:
- Dashboard: `http://localhost:8000/dashboard/`
- API: `http://localhost:8000/dashboard/api/performance`
- System: `http://localhost:8000/dashboard/api/system`

### Automated Validation System

Monitor TTS quality with Whisper-based validation:

- **🔍 Quality Assessment** - Automatic pronunciation, prosody, and naturalness scoring
- **🧪 Automated Testing** - Comprehensive test suites with diverse text samples
- **🔧 Self-Debugging** - Automatic issue detection and root cause analysis
- **📊 Health Monitoring** - Real-time system health scoring and trend analysis

Access validation features via:
- Dashboard: `http://localhost:8000/dashboard/` (Validation section)
- Health API: `http://localhost:8000/dashboard/api/validation/health`
- Test API: `http://localhost:8000/dashboard/api/validation/test`

## ✨ Key Features

- **🏃‍♂️ Lightning Fast**: RTF < 0.5 on CPU-only hardware (4-core, 4GB RAM)
- **🔄 OpenAI Compatible**: Drop-in replacement for OpenAI TTS API
- **🎭 Voice Cloning**: Create custom voices from 10-30 second audio samples
- **💾 Resource Efficient**: < 2GB memory usage, optimized for constrained environments
- **🐳 Easy Deployment**: Single command setup with Docker support
- **🔊 High Quality**: MOS > 3.8, professional-grade audio output
- **🌐 Multiple Formats**: MP3, WAV, FLAC, Opus, AAC support
- **📡 Streaming**: Real-time audio streaming for long text
- **⚙️ Flexible Configuration**: JSON config files with CLI override support
- **🎵 Interactive Dashboard**: Web UI with audio playback and live metrics
- **📊 Real-Time Monitoring**: Live performance metrics and system monitoring
- **🎛️ Audio Controls**: Built-in audio player with download capabilities
- **🔍 Automated Validation**: Whisper-based quality assurance and self-debugging
- **🧪 Testing Pipeline**: Comprehensive automated testing with quality scoring

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

#### Localhost Access (Default)
```bash
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openaudio-s1-mini",
    "input": "Hello, this is JabberTTS speaking!",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

#### Network Access (Replace `YOUR_SERVER_IP` with actual IP)
```bash
curl -X POST "http://YOUR_SERVER_IP:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openaudio-s1-mini",
    "input": "Hello, this is JabberTTS speaking from the network!",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Voice Examples

#### All Supported Voices
```bash
# Alloy - Balanced, neutral voice
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is the Alloy voice.", "voice": "alloy", "response_format": "mp3"}' \
  --output alloy_voice.mp3

# Echo - Clear, articulate voice
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is the Echo voice.", "voice": "echo", "response_format": "mp3"}' \
  --output echo_voice.mp3

# Fable - Warm, storytelling voice
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is the Fable voice.", "voice": "fable", "response_format": "mp3"}' \
  --output fable_voice.mp3

# Onyx - Deep, authoritative voice
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is the Onyx voice.", "voice": "onyx", "response_format": "mp3"}' \
  --output onyx_voice.mp3

# Nova - Bright, energetic voice
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is the Nova voice.", "voice": "nova", "response_format": "mp3"}' \
  --output nova_voice.mp3

# Shimmer - Gentle, soothing voice
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is the Shimmer voice.", "voice": "shimmer", "response_format": "mp3"}' \
  --output shimmer_voice.mp3
```

### Audio Format Examples

#### All Supported Formats
```bash
# MP3 (Default) - Good compression, widely supported
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "MP3 format example", "voice": "alloy", "response_format": "mp3"}' \
  --output example.mp3

# WAV - Uncompressed, high quality
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "WAV format example", "voice": "alloy", "response_format": "wav"}' \
  --output example.wav

# FLAC - Lossless compression
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "FLAC format example", "voice": "alloy", "response_format": "flac"}' \
  --output example.flac

# Opus - Efficient compression for streaming
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "Opus format example", "voice": "alloy", "response_format": "opus"}' \
  --output example.opus

# AAC - Good compression, mobile-friendly
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "AAC format example", "voice": "alloy", "response_format": "aac"}' \
  --output example.aac

# PCM - Raw audio data
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "PCM format example", "voice": "alloy", "response_format": "pcm"}' \
  --output example.pcm
```

### Speed Variations

```bash
# Very slow (0.25x)
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is very slow speech.", "voice": "alloy", "speed": 0.25}' \
  --output very_slow.mp3

# Slow (0.5x)
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is slow speech.", "voice": "alloy", "speed": 0.5}' \
  --output slow.mp3

# Normal (1.0x - default)
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is normal speed speech.", "voice": "alloy", "speed": 1.0}' \
  --output normal.mp3

# Fast (1.5x)
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is fast speech.", "voice": "alloy", "speed": 1.5}' \
  --output fast.mp3

# Very fast (2.0x)
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is very fast speech.", "voice": "alloy", "speed": 2.0}' \
  --output very_fast.mp3

# Maximum speed (4.0x)
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "This is maximum speed speech.", "voice": "alloy", "speed": 4.0}' \
  --output max_speed.mp3
```

### Text Complexity Examples

```bash
# Short text
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "Hello world!", "voice": "alloy"}' \
  --output short.mp3

# Medium text with punctuation
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "Welcome to JabberTTS! This is a medium-length text with punctuation, numbers like 123, and various symbols.", "voice": "alloy"}' \
  --output medium.mp3

# Long text with complex content
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "JabberTTS is a state-of-the-art text-to-speech system that converts written text into natural-sounding speech. It supports multiple voices, audio formats, and speed adjustments. The system is optimized for real-time performance with RTF values under 0.5 on CPU-only hardware. Technical specifications include support for sample rates up to 48kHz, multiple audio codecs, and advanced phonetic processing.", "voice": "alloy"}' \
  --output long.mp3
```

### List Available Voices

```bash
# Get all available voices
curl -X GET "http://localhost:8000/v1/voices" \
  -H "Content-Type: application/json"

# Example response:
# {
#   "voices": [
#     {"id": "alloy", "name": "Alloy", "description": "A balanced, neutral voice", "type": "built-in"},
#     {"id": "echo", "name": "Echo", "description": "A clear, articulate voice", "type": "built-in"},
#     ...
#   ]
# }
```

### Health Check and Monitoring

```bash
# Basic health check
curl -X GET "http://localhost:8000/health"

# Example response:
# {"status": "healthy", "version": "0.1.0", "service": "jabbertts"}

# Root endpoint with service info
curl -X GET "http://localhost:8000/"

# Example response:
# {
#   "service": "JabberTTS",
#   "version": "0.1.0",
#   "description": "Fast, efficient Text-to-Speech API with OpenAI compatibility",
#   "docs": "/docs",
#   "health": "/health"
# }
```

## 🎛️ Web Dashboard

JabberTTS includes a modern web-based dashboard for testing and demonstrating TTS capabilities with enhanced audio quality features.

### Accessing the Dashboard

```bash
# Start the server
uv run python app.py

# Open your browser and navigate to:
http://localhost:8000/dashboard/
```

### Dashboard Features

#### 🎵 **Text-to-Speech Generator with Audio Playback**
- **Interactive Text Input**: Large text area with sample content
- **Voice Selection**: Visual grid showing all 6 available voices (Alloy, Echo, Fable, Onyx, Nova, Shimmer)
- **Audio Format Selection**: Dropdown with all supported formats (MP3, WAV, FLAC, Opus, AAC, PCM)
- **Speed Control**: Interactive slider from 0.25x to 4.0x speed
- **Real-time Generation**: Progress feedback with loading animations
- **🔊 Automatic Audio Playback**: Generated audio plays automatically in the browser
- **🎛️ Audio Controls**: Play, pause, stop, and volume controls
- **💾 Download Support**: One-click download of generated audio files
- **📊 Enhanced Results**: RTF, generation time, and detailed audio metrics

#### 📊 **Real-Time System Metrics**
- **Live Performance Metrics**: Real-time RTF, response time, memory, and CPU usage
- **Request Statistics**: Total requests, success rate, error tracking
- **System Resources**: Actual CPU and memory usage from psutil
- **Uptime Monitoring**: System availability and service health
- **Color-coded Indicators**: Visual status indicators for performance thresholds
- **Auto-refresh**: Updates every 30 seconds with live data

#### 🔧 **Audio Quality Information**
- **Quality Preset Display**: Current quality settings (Low, Standard, High, Ultra)
- **Enhancement Features**: Visual indicators for noise reduction, compression, normalization
- **Sample Rate Information**: Current and target sample rates
- **Feature Status**: Real-time status of audio enhancement features

### Dashboard API Endpoints

```bash
# Get system status
curl -X GET "http://localhost:8000/dashboard/api/status"

# Get available voices with preview text
curl -X GET "http://localhost:8000/dashboard/api/voices"

# Get performance metrics
curl -X GET "http://localhost:8000/dashboard/api/performance"

# Generate speech via dashboard API
curl -X POST "http://localhost:8000/dashboard/generate" \
  -F "text=Hello from the dashboard!" \
  -F "voice=alloy" \
  -F "format=mp3" \
  -F "speed=1.0"
```

### Network Access

When running with network access (`JABBERTTS_HOST=0.0.0.0`), the dashboard is accessible from other machines:

```bash
# Access from another machine (replace with actual server IP)
http://YOUR_SERVER_IP:8000/dashboard/
```

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8001/v1/audio/speech",
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

### Server Configuration

```bash
# Network configuration
export JABBERTTS_HOST=0.0.0.0          # 0.0.0.0 for network access, 127.0.0.1 for localhost only
export JABBERTTS_PORT=8000              # Server port
export JABBERTTS_WORKERS=1              # Number of worker processes
export JABBERTTS_AUTO_PORT=true         # Automatically find available port if configured port is in use

# Development settings
export JABBERTTS_RELOAD=false           # Enable auto-reload for development
export JABBERTTS_DEBUG=false            # Enable debug mode
export JABBERTTS_LOG_LEVEL=INFO         # Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Model Configuration

```bash
export JABBERTTS_MODEL_PATH=/models/openaudio-s1-mini.onnx
export JABBERTTS_MODEL_NAME=speecht5
export JABBERTTS_MAX_TEXT_LENGTH=4096
```

### Performance Configuration

```bash
export JABBERTTS_MAX_CONCURRENT_REQUESTS=100
export JABBERTTS_REQUEST_TIMEOUT=30
export JABBERTTS_INFERENCE_TIMEOUT=20
```

### Audio Configuration

```bash
export JABBERTTS_DEFAULT_VOICE=alloy
export JABBERTTS_DEFAULT_FORMAT=mp3
export JABBERTTS_DEFAULT_SPEED=1.0
export JABBERTTS_SAMPLE_RATE=24000
```

### Security Configuration

```bash
# Authentication (optional)
export JABBERTTS_API_KEY=your-secret-key
export JABBERTTS_RATE_LIMIT=100         # requests per minute
export JABBERTTS_ENABLE_DOCS=true       # Enable API documentation
```

### Port Binding Features

JabberTTS includes intelligent port binding with automatic fallback:

```bash
# Automatic port discovery
export JABBERTTS_AUTO_PORT=true
uv run python app.py
# If port 8000 is in use, automatically finds and uses port 8001, 8002, etc.

# Manual port selection with fallback
uv run python app.py
# If port 8000 is in use, prompts: "Port 8000 is in use. Use port 8001 instead? (y/N):"

# Custom port
export JABBERTTS_PORT=8080
uv run python app.py
# Uses port 8080 if available
```

### Network Access Configuration

#### Localhost Only (Secure)
```bash
export JABBERTTS_HOST=127.0.0.1
# API accessible only from the same machine
```

#### Network Access (Development/Testing)
```bash
export JABBERTTS_HOST=0.0.0.0
# API accessible from other machines on the network
# ⚠️ WARNING: Only use in trusted networks or with proper authentication
```

## 🔒 Security Considerations

### Network Exposure Warnings

When configuring JabberTTS for network access (`JABBERTTS_HOST=0.0.0.0`), be aware of the following security implications:

#### ⚠️ **IMPORTANT SECURITY WARNINGS**

1. **Unrestricted Access**: The API will be accessible from any machine that can reach your server
2. **No Built-in Authentication**: By default, there's no authentication required
3. **Resource Consumption**: Malicious users could consume server resources
4. **Data Privacy**: Text inputs are processed and temporarily stored in memory

#### 🛡️ **Security Best Practices**

**For Development/Testing:**
```bash
# Use localhost binding when possible
export JABBERTTS_HOST=127.0.0.1

# Use non-standard ports
export JABBERTTS_PORT=8080

# Enable API key authentication
export JABBERTTS_API_KEY=your-secure-random-key-here
```

**For Production Deployment:**
```bash
# Use a reverse proxy (nginx, Apache) with SSL/TLS
# Configure firewall rules to restrict access
# Implement rate limiting
export JABBERTTS_RATE_LIMIT=10  # requests per minute per IP

# Disable API documentation in production
export JABBERTTS_ENABLE_DOCS=false
```

**Firewall Configuration:**
```bash
# Ubuntu/Debian - Allow specific IP ranges only
sudo ufw allow from 192.168.1.0/24 to any port 8000
sudo ufw deny 8000

# Block all external access, allow only local network
sudo iptables -A INPUT -p tcp --dport 8000 -s 192.168.0.0/16 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j DROP
```

**Reverse Proxy Setup (nginx example):**
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Rate limiting
        limit_req zone=api burst=10 nodelay;
    }
}
```

#### 🔍 **Monitoring and Logging**

Enable comprehensive logging for security monitoring:
```bash
export JABBERTTS_ACCESS_LOG=true
export JABBERTTS_LOG_LEVEL=INFO

# Monitor access patterns
tail -f /var/log/jabbertts/access.log | grep -E "(POST|GET)"
```

#### 🚫 **What NOT to Do**

- ❌ Don't expose JabberTTS directly to the internet without authentication
- ❌ Don't use default ports (8000) in production
- ❌ Don't run as root user
- ❌ Don't ignore rate limiting in multi-user environments
- ❌ Don't store sensitive data in text inputs

#### ✅ **Recommended Production Setup**

1. **Use Docker with non-root user**
2. **Deploy behind a reverse proxy with SSL/TLS**
3. **Implement API key authentication**
4. **Configure rate limiting and request size limits**
5. **Monitor access logs and resource usage**
6. **Use private networks or VPN for internal access**
7. **Regular security updates and dependency scanning**

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

## 🔧 Troubleshooting

### Port Binding Issues

#### Problem: "Address already in use" error
```bash
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```

**Solutions:**

1. **Use automatic port discovery:**
   ```bash
   export JABBERTTS_AUTO_PORT=true
   uv run python app.py
   ```

2. **Use a different port:**
   ```bash
   export JABBERTTS_PORT=8080
   uv run python app.py
   ```

3. **Find what's using the port:**
   ```bash
   # Check what's using port 8000
   sudo netstat -tulpn | grep :8000
   # or
   sudo lsof -i :8000
   ```

4. **Kill the process using the port:**
   ```bash
   # Find the process ID
   sudo lsof -t -i:8000
   # Kill the process (replace PID with actual process ID)
   sudo kill -9 PID
   ```

### Network Connectivity Issues

#### Problem: Cannot access API from other machines

**Check server binding:**
```bash
# Ensure server is bound to 0.0.0.0, not 127.0.0.1
export JABBERTTS_HOST=0.0.0.0
uv run python app.py
```

**Check firewall settings:**
```bash
# Ubuntu/Debian
sudo ufw allow 8000
sudo ufw status

# CentOS/RHEL
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

**Test connectivity:**
```bash
# From another machine, replace SERVER_IP with actual IP
curl -X GET "http://SERVER_IP:8000/health"
```

### Audio Generation Issues

#### Problem: TTS request fails or returns empty audio

**Check model availability:**
```bash
curl -X GET "http://localhost:8000/v1/models"
```

**Verify text input:**
```bash
# Test with simple text first
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "test", "voice": "alloy"}' \
  --output test.mp3
```

**Check logs for errors:**
```bash
# Run with debug logging
export JABBERTTS_LOG_LEVEL=DEBUG
uv run python app.py
```

#### Problem: Poor audio quality

**Try different voices:**
```bash
# Test all voices to find the best one for your use case
for voice in alloy echo fable onyx nova shimmer; do
  curl -X POST "http://localhost:8000/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"openaudio-s1-mini\", \"input\": \"Testing voice quality\", \"voice\": \"$voice\"}" \
    --output "test_$voice.mp3"
done
```

**Use higher quality formats:**
```bash
# Use FLAC for lossless audio
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "High quality test", "voice": "alloy", "response_format": "flac"}' \
  --output high_quality.flac
```

### Performance Issues

#### Problem: Slow response times

**Check system resources:**
```bash
# Monitor CPU and memory usage
htop
# or
top
```

**Optimize configuration:**
```bash
# Reduce concurrent requests if system is overloaded
export JABBERTTS_MAX_CONCURRENT_REQUESTS=50
export JABBERTTS_REQUEST_TIMEOUT=60
export JABBERTTS_INFERENCE_TIMEOUT=30
```

**Performance benchmarking:**
```bash
# Test single request performance
time curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "openaudio-s1-mini", "input": "Performance test", "voice": "alloy"}' \
  --output perf_test.mp3

# Load testing with multiple concurrent requests
uv run locust -f tests/load/locustfile.py --host=http://localhost:8000 -u 10 -r 2 -t 60s
```

### Expected Performance Metrics

| Metric | Target Value | Description |
|--------|--------------|-------------|
| RTF (Real-Time Factor) | < 0.5 | Time to generate audio / Audio duration |
| Response Time | < 3s | For typical sentences (10-50 words) |
| Throughput | > 10 req/s | Concurrent requests on modern CPU |
| Memory Usage | < 2GB | Peak memory consumption |

**RTF Calculation Example:**
```bash
# If it takes 1.5 seconds to generate 3 seconds of audio:
# RTF = 1.5s / 3.0s = 0.5 (Good performance)
```

### Common Error Messages

#### "Model not found"
```bash
# Ensure model files are available
ls -la models/
# Download models if missing (see installation section)
```

#### "Phonemizer not available"
```bash
# Install espeak-ng
sudo apt-get install espeak-ng  # Ubuntu/Debian
sudo yum install espeak-ng      # CentOS/RHEL
brew install espeak-ng          # macOS
```

#### "FFmpeg not found"
```bash
# Install FFmpeg
sudo apt-get install ffmpeg     # Ubuntu/Debian
sudo yum install ffmpeg         # CentOS/RHEL
brew install ffmpeg             # macOS
```

### Debug Mode

Enable comprehensive debugging:

```bash
export JABBERTTS_DEBUG=true
export JABBERTTS_LOG_LEVEL=DEBUG
export JABBERTTS_PROFILING=true
uv run python app.py
```

This will provide detailed logs for:
- Request processing
- Model loading and inference
- Audio processing pipeline
- Performance metrics
- Error stack traces

### Getting Help

If you encounter issues not covered here:

1. **Check the logs** with debug mode enabled
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - JabberTTS version
   - Operating system and version
   - Python version
   - Complete error message
   - Steps to reproduce
   - Configuration used

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
