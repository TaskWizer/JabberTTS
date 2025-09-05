# JabberTTS API - Technical Specifications

## 1. Executive Summary

JabberTTS is a high-performance, OpenAI-compatible Text-to-Speech API designed for CPU-only operation with Real-Time Factor (RTF) < 0.5 on constrained hardware. The system leverages the optimized OpenAudio S1-mini model with ONNX Runtime acceleration to deliver near-instant response times while maintaining high audio quality.

## 2. Functional Specifications

### 2.1 Core API Endpoints

#### 2.1.1 Speech Generation Endpoint
- **Endpoint**: `POST /v1/audio/speech`
- **Purpose**: Generate speech audio from input text
- **OpenAI Compatibility**: 100% compatible with OpenAI Audio API schema

**Request Schema:**
```json
{
  "model": "openaudio-s1-mini",
  "input": "Text to convert to speech",
  "voice": "alloy|echo|fable|onyx|nova|shimmer|custom",
  "response_format": "mp3|opus|aac|flac|wav|pcm",
  "speed": 0.25-4.0
}
```

**Response**: Streaming audio data in specified format

#### 2.1.2 Health Check Endpoint
- **Endpoint**: `GET /health`
- **Purpose**: System health monitoring
- **Response**: JSON with system status, model load status, and performance metrics

#### 2.1.3 Metrics Endpoint
- **Endpoint**: `GET /metrics`
- **Purpose**: Prometheus-compatible metrics for monitoring
- **Response**: Metrics in Prometheus format

### 2.2 Voice Cloning Capabilities

#### 2.2.1 Voice Upload
- **Endpoint**: `POST /v1/voices`
- **Purpose**: Upload reference audio for voice cloning
- **Requirements**: 
  - Audio length: 10-30 seconds
  - Format: WAV, MP3, FLAC
  - Sample rate: 16kHz-48kHz
  - Quality: Clear speech, minimal background noise

#### 2.2.2 Voice Management
- **Endpoint**: `GET /v1/voices` - List available voices
- **Endpoint**: `DELETE /v1/voices/{voice_id}` - Remove custom voice

## 3. Technical Specifications

### 3.1 Performance Requirements

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Real-Time Factor (RTF)** | < 0.5 | Inference time / audio duration |
| **Memory Usage** | < 2GB RAM | Peak RSS during inference |
| **First Token Latency** | < 1s | Time to first audio chunk |
| **Throughput** | 10+ concurrent requests | Load testing with locust |
| **Audio Quality (MOS)** | > 3.8 | Subjective evaluation |
| **Word Error Rate (WER)** | < 5% | ASR transcription accuracy |

### 3.2 Hardware Requirements

#### 3.2.1 Minimum Requirements
- **CPU**: 4-core Intel i5-8300H or equivalent (AVX2 support)
- **RAM**: 4GB available memory
- **Storage**: 2GB for model files
- **Network**: 100Mbps for concurrent requests

#### 3.2.2 Recommended Requirements
- **CPU**: 8-core Intel i7 or AMD Ryzen 7
- **RAM**: 8GB available memory
- **Storage**: SSD with 5GB available space
- **Network**: 1Gbps for high-throughput scenarios

### 3.3 Model Specifications

#### 3.3.1 OpenAudio S1-mini Optimized
- **Original Size**: ~1.2GB (FP16)
- **Optimized Size**: 300-400MB (Mixed-bit quantization)
- **Quantization Strategy**: 
  - 2-3 bits: Attention layers
  - 4 bits: Linear layers and embeddings
  - 8 bits: Output layers and codec
- **Context Window**: Reduced for CPU efficiency
- **Languages**: Multilingual support (English primary)

#### 3.3.2 Audio Processing
- **Sample Rate**: 24kHz output
- **Bit Depth**: 16-bit
- **Encoding**: MP3 96kbps (default), supports multiple formats
- **Streaming**: Chunked generation and encoding

### 3.4 Architecture Specifications

#### 3.4.1 Technology Stack
- **API Framework**: FastAPI 0.104+
- **Inference Engine**: ONNX Runtime 1.16+ (CPUExecutionProvider)
- **Text Preprocessing**: eSpeak-NG via phonemizer
- **Audio Encoding**: FFmpeg via ffmpeg-python
- **Async Runtime**: uvicorn with asyncio
- **Dependency Management**: uv for Python package management

#### 3.4.2 Optimization Pipeline
```
PyTorch Model → Pruning → Mixed-Bit Quantization → ONNX Conversion → ONNX Runtime Optimization → Production Deployment
```

#### 3.4.3 Request Processing Flow
```
Client Request → FastAPI → Input Validation → Text Preprocessing → Model Inference → Audio Encoding → Streaming Response
```

## 4. API Contract Specifications

### 4.1 Request/Response Formats

#### 4.1.1 Content Types
- **Request**: `application/json`
- **Response**: `audio/mpeg`, `audio/wav`, `audio/ogg`, etc.
- **Errors**: `application/json`

#### 4.1.2 Error Handling
```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error|api_error|overloaded_error",
    "param": "parameter_name",
    "code": "error_code"
  }
}
```

#### 4.1.3 Rate Limiting
- **Default**: 100 requests/minute per IP
- **Configurable**: Via environment variables
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`

### 4.2 Authentication & Security

#### 4.2.1 API Key Authentication
- **Header**: `Authorization: Bearer <api_key>`
- **Optional**: Can be disabled for local deployment
- **Key Management**: Environment variable configuration

#### 4.2.2 Input Validation
- **Text Length**: Maximum 4096 characters
- **Voice Parameter**: Validated against available voices
- **Speed Parameter**: Range 0.25-4.0
- **Format Parameter**: Validated against supported formats

## 5. Quality Assurance Specifications

### 5.1 Testing Requirements

#### 5.1.1 Unit Testing
- **Coverage**: Minimum 80%
- **Framework**: pytest
- **Scope**: All core functions, API endpoints, model inference

#### 5.1.2 Integration Testing
- **API Testing**: Full request/response cycle
- **Model Testing**: Audio generation quality
- **Performance Testing**: RTF and memory benchmarks

#### 5.1.3 Load Testing
- **Tool**: locust
- **Scenarios**: 1, 10, 50, 100 concurrent users
- **Duration**: 10-minute sustained load tests
- **Metrics**: Response time, error rate, throughput

### 5.2 Monitoring & Observability

#### 5.2.1 Metrics Collection
- **Request Metrics**: Count, duration, error rate
- **Model Metrics**: Inference time, memory usage
- **System Metrics**: CPU, memory, disk usage

#### 5.2.2 Logging
- **Format**: Structured JSON logging
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Content**: Request ID, user context, performance data

## 6. Deployment Specifications

### 6.1 Container Specifications
- **Base Image**: python:3.11-slim
- **Size Target**: < 2GB final image
- **Multi-stage Build**: Separate build and runtime stages
- **Health Checks**: Built-in container health monitoring

### 6.2 Configuration Management
- **Environment Variables**: All configuration externalized
- **Secrets**: API keys, model paths via secure mounting
- **Defaults**: Sensible defaults for development

### 6.3 Scalability
- **Horizontal Scaling**: Stateless design for load balancing
- **Resource Limits**: Configurable CPU/memory limits
- **Queue Management**: Request queuing for burst handling

## 7. Compliance & Standards

### 7.1 API Standards
- **OpenAPI 3.0**: Complete API documentation
- **REST Principles**: Stateless, cacheable, uniform interface
- **HTTP Standards**: Proper status codes, headers

### 7.2 Code Standards
- **Python**: PEP 8 compliance
- **Type Hints**: Full type annotation
- **Documentation**: Comprehensive docstrings
- **Linting**: black, flake8, mypy

### 7.3 Security Standards
- **Input Sanitization**: All user inputs validated
- **Error Handling**: No sensitive information in errors
- **Dependencies**: Regular security updates
- **Audit Trail**: Request logging for security monitoring
