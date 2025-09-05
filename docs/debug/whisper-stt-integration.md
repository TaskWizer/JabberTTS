# Whisper STT Integration for JabberTTS

## Overview

The Whisper STT (Speech-to-Text) integration provides comprehensive audio intelligibility testing and debugging capabilities for JabberTTS. This system uses OpenAI's Whisper model to transcribe generated TTS audio back to text, enabling accurate measurement of audio intelligibility and quality.

## Features

### 1. Dashboard Debug Endpoints

#### `/debug/transcribe` - Audio File Transcription
Upload audio files for transcription analysis with accuracy metrics.

**Request:**
```bash
curl -X POST "http://localhost:8002/dashboard/debug/transcribe" \
  -F "audio_file=@test.wav" \
  -F "original_text=Hello world"
```

**Response:**
```json
{
  "success": true,
  "filename": "test.wav",
  "transcription": "Hello world",
  "segments": [...],
  "transcription_info": {
    "transcription_time": 0.95,
    "detected_language": "en",
    "language_probability": 0.99
  },
  "original_text": "Hello world",
  "accuracy_metrics": {
    "overall_accuracy": 95.2,
    "wer": 0.048,
    "cer": 0.023
  },
  "audio_analysis": {
    "duration": 1.5,
    "sample_rate": 16000,
    "quality_metrics": {...}
  }
}
```

#### `/debug/generate-and-transcribe` - End-to-End Testing
Generate TTS audio and immediately transcribe it for comprehensive testing.

**Request:**
```bash
curl -X POST "http://localhost:8002/dashboard/debug/generate-and-transcribe" \
  -F "text=The quick brown fox jumps over the lazy dog" \
  -F "voice=alloy" \
  -F "format=wav" \
  -F "speed=1.0"
```

**Response:**
```json
{
  "success": true,
  "original_text": "The quick brown fox jumps over the lazy dog",
  "generation_info": {
    "voice": "alloy",
    "format": "wav",
    "speed": 1.0,
    "rtf": 0.45,
    "inference_time": 2.1,
    "audio_duration": 4.7
  },
  "transcription_result": {
    "transcription": "The quick brown fox jumps over the lazy dog",
    "accuracy_metrics": {
      "overall_accuracy": 98.5,
      "wer": 0.015,
      "cer": 0.008
    }
  },
  "audio_data": "data:audio/wav;base64,..."
}
```

#### `/debug/audio-analysis` - Comprehensive Audio Analysis
Generate detailed audio analysis with waveform, spectrogram, and phoneme data.

**Request:**
```bash
curl -X POST "http://localhost:8002/dashboard/debug/audio-analysis" \
  -F "text=Hello world" \
  -F "voice=fable" \
  -F "format=wav" \
  -F "include_waveform=true" \
  -F "include_spectrogram=true" \
  -F "include_phonemes=true"
```

**Response:**
```json
{
  "success": true,
  "original_text": "Hello world",
  "generation_info": {...},
  "phoneme_analysis": {
    "original_text": "Hello world",
    "phonemized_text": "həlˈoʊ wˈɜːld",
    "phoneme_count": 2,
    "complexity_score": 0.214
  },
  "waveform": {
    "amplitude": [...],
    "time": [...],
    "duration": 1.2,
    "sample_rate": 16000
  },
  "spectrogram": {
    "frequencies": [...],
    "times": [...],
    "magnitude_db": [[...]]
  },
  "quality_analysis": {
    "overall_score": 95.8,
    "metrics": {...}
  }
}
```

### 2. Dashboard Web Interface

The dashboard provides an intuitive web interface for audio debugging:

#### Debug Transcription System
- **Upload Audio Files**: Drag-and-drop interface for audio file upload
- **Original Text Input**: Optional field for accuracy comparison
- **Real-time Results**: Side-by-side comparison of original vs transcribed text
- **Accuracy Metrics**: Visual display of WER, CER, and accuracy percentages

#### Generate & Transcribe
- **Text Input**: Multi-line text area for TTS generation
- **Voice Selection**: Dropdown for all available voices
- **Format Options**: WAV, MP3, FLAC, OPUS support
- **Instant Playback**: Generated audio with embedded player
- **Live Metrics**: Real-time RTF, accuracy, and quality scores

#### Audio Waveform & Spectral Analysis
- **Comprehensive Analysis**: Waveform, spectrogram, and phoneme visualization
- **Interactive Controls**: Toggle waveform, spectrogram, and phoneme analysis
- **Quality Metrics**: Detailed quality assessment with scoring
- **Export Options**: JSON export for detailed analysis

## Metrics and Scoring

### Accuracy Metrics

#### Word Error Rate (WER)
```
WER = (S + D + I) / N
```
Where:
- S = Substitutions
- D = Deletions  
- I = Insertions
- N = Total words in reference

**Thresholds:**
- Excellent: WER ≤ 0.05 (5%)
- Good: WER ≤ 0.15 (15%)
- Poor: WER > 0.15

#### Character Error Rate (CER)
```
CER = (S + D + I) / N
```
Similar to WER but at character level.

**Thresholds:**
- Excellent: CER ≤ 0.05 (5%)
- Good: CER ≤ 0.15 (15%)
- Poor: CER > 0.15

#### Overall Accuracy
```
Accuracy = (1 - WER) × 100%
```

**Thresholds:**
- Excellent: ≥95%
- Good: ≥85%
- Acceptable: ≥70%
- Poor: <70%

### Quality Metrics

#### Technical Quality
- **Overall Quality**: Composite score (0-100%)
- **Naturalness**: Human-like quality assessment
- **Clarity**: Audio clarity and intelligibility
- **Consistency**: Temporal consistency across audio

#### Performance Metrics
- **RTF (Real-Time Factor)**: Inference time / audio duration
- **Inference Time**: Total generation time
- **Audio Duration**: Length of generated audio

## Usage Examples

### 1. Basic Intelligibility Testing
```python
from jabbertts.scripts.validate_intelligibility_framework import validate_intelligibility_framework

# Run comprehensive intelligibility validation
results = await validate_intelligibility_framework()
print(f"Average accuracy: {results['avg_accuracy']:.1f}%")
```

### 2. Automated Testing Pipeline
```python
from tests.test_intelligibility import TestIntelligibilityFramework

# Run automated test suite
framework = TestIntelligibilityFramework()
results = await framework.test_comprehensive_intelligibility_suite()
```

### 3. Dashboard Integration
Access the web interface at `http://localhost:8002/dashboard/` and navigate to the "Debug Transcription System" section.

## Configuration

### Whisper Model Selection
The system supports different Whisper model sizes:
- `tiny`: Fastest, least accurate
- `base`: Balanced speed/accuracy (default)
- `small`: Better accuracy, slower
- `medium`: High accuracy, much slower
- `large`: Best accuracy, very slow

Configure in `jabbertts/validation/whisper_validator.py`:
```python
whisper_validator = get_whisper_validator("base")  # Change model size here
```

### Quality Thresholds
Adjust quality thresholds in test files:
```python
baseline = QualityBaseline(
    min_transcription_accuracy=95.0,
    max_word_error_rate=0.05,
    min_overall_quality=85.0
)
```

## Troubleshooting

### Common Issues

#### 1. Low Transcription Accuracy
- **Symptoms**: Accuracy <50%, high WER/CER
- **Causes**: Audio intelligibility issues, model problems
- **Solutions**: Check audio pipeline, verify model configuration

#### 2. Whisper Model Loading Errors
- **Symptoms**: Import errors, model not found
- **Solutions**: Ensure `faster-whisper` is installed, check model path

#### 3. Audio Format Issues
- **Symptoms**: Transcription fails, format errors
- **Solutions**: Ensure audio is in supported format (WAV recommended)

### Debug Commands
```bash
# Test Whisper integration
python jabbertts/scripts/validate_intelligibility_framework.py

# Test perceptual quality
python jabbertts/scripts/simple_perceptual_test.py

# Test dashboard endpoints
python jabbertts/scripts/test_whisper_dashboard_simple.py
```

## API Reference

### WhisperValidator Class
```python
class WhisperValidator:
    def validate_tts_output(self, original_text: str, audio_data: bytes, sample_rate: int) -> Dict
    def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> Dict
    def calculate_accuracy_metrics(self, original: str, transcribed: str) -> Dict
```

### Dashboard Routes
- `POST /dashboard/debug/transcribe`
- `POST /dashboard/debug/generate-and-transcribe`  
- `POST /dashboard/debug/audio-analysis`

## Best Practices

1. **Use WAV format** for highest accuracy
2. **Test with diverse text samples** for comprehensive validation
3. **Monitor accuracy trends** over time for regression detection
4. **Combine with technical metrics** for complete quality assessment
5. **Regular validation** during development to catch issues early

## Integration with CI/CD

Add automated intelligibility testing to your CI pipeline:
```yaml
- name: Run Intelligibility Tests
  run: |
    python -m pytest tests/test_intelligibility.py -v
    python jabbertts/scripts/validate_intelligibility_framework.py
```
