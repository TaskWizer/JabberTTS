# Perceptual Quality Metrics Framework

## Overview

The Perceptual Quality Metrics Framework provides comprehensive assessment of TTS audio quality beyond traditional technical metrics. It evaluates human-like characteristics including prosody, rhythm, emotional expression, and overall human-likeness to ensure generated speech sounds natural and engaging.

## Framework Components

### 1. Perceptual Quality Analyzer

The `PerceptualQualityAnalyzer` class provides advanced analysis capabilities:

```python
from tests.test_perceptual_quality import PerceptualQualityAnalyzer

analyzer = PerceptualQualityAnalyzer()
metrics = await analyzer.analyze_comprehensive_quality(
    text="Hello world",
    voice="alloy", 
    inference_engine=engine,
    audio_processor=processor
)
```

### 2. Comprehensive Metrics

#### Core Intelligibility Metrics
- **Transcription Accuracy**: Whisper STT accuracy percentage
- **Word Error Rate (WER)**: Word-level transcription errors
- **Character Error Rate (CER)**: Character-level transcription errors

#### Technical Quality Metrics  
- **Overall Quality**: Composite technical quality score
- **Naturalness Score**: Human-like quality assessment
- **Clarity Score**: Audio clarity and intelligibility
- **Consistency Score**: Temporal consistency across audio

#### Perceptual Quality Metrics (NEW)
- **Prosody Score**: Speech rhythm and intonation quality
- **Rhythm Score**: Temporal regularity and flow
- **Emotional Expression**: Spectral variation and expressiveness
- **Human Likeness**: Composite human-like quality score

#### Performance Metrics
- **RTF (Real-Time Factor)**: Generation efficiency
- **Inference Time**: Total processing time
- **Audio Duration**: Length of generated speech

### 3. Quality Baselines and Thresholds

```python
@dataclass
class QualityBaseline:
    min_transcription_accuracy: float = 95.0    # Excellent intelligibility
    max_word_error_rate: float = 0.05           # <5% word errors
    max_character_error_rate: float = 0.05      # <5% character errors
    min_overall_quality: float = 85.0           # High technical quality
    min_naturalness: float = 80.0               # Natural sounding
    min_clarity: float = 85.0                   # Clear and intelligible
    min_prosody: float = 70.0                   # Good prosody
    min_human_likeness: float = 75.0            # Human-like quality
    max_rtf: float = 0.5                        # Real-time performance
```

## Perceptual Analysis Methods

### 1. Prosody and Rhythm Analysis

Evaluates speech timing, stress patterns, and intonation:

```python
def analyze_prosody_and_rhythm(audio_data, sample_rate, text):
    # Speech rate analysis (words per minute)
    # Amplitude variation analysis (prosodic contour)  
    # Rhythm regularity assessment
    return prosody_score, rhythm_score
```

**Key Features:**
- **Speech Rate Optimization**: Targets 150-180 WPM ideal range
- **Amplitude Variation**: Measures prosodic contour naturalness
- **Rhythm Consistency**: Evaluates temporal regularity
- **Stress Pattern Detection**: Identifies emphasis and timing

### 2. Emotional Expression Analysis

Assesses emotional content and expressiveness:

```python
def analyze_emotional_expression(audio_data, sample_rate, text):
    # Spectral centroid analysis for brightness
    # Spectral variation for emotional content
    # Text-based emotional context detection
    return emotional_expression_score
```

**Key Features:**
- **Spectral Analysis**: Frequency content variation
- **Brightness Assessment**: Spectral centroid tracking
- **Contextual Awareness**: Text-based emotional detection
- **Expression Variation**: Dynamic range evaluation

### 3. Human-Likeness Calculation

Composite score combining multiple perceptual factors:

```python
def calculate_human_likeness(metrics):
    weights = {
        'naturalness_score': 0.3,
        'prosody_score': 0.25, 
        'clarity_score': 0.2,
        'emotional_expression': 0.15,
        'rhythm_score': 0.1
    }
    return weighted_average(metrics, weights)
```

## Testing Framework

### 1. Automated Test Suite

```python
class TestPerceptualQualityFramework:
    TEST_CASES = [
        # Simple texts
        ("Hello world.", "alloy", "simple"),
        ("Good morning.", "fable", "simple"),
        
        # Medium complexity  
        ("The weather today is sunny and warm.", "onyx", "medium"),
        
        # Complex texts
        ("Neural networks enable advanced AI applications.", "alloy", "complex"),
    ]
```

### 2. Regression Testing

Prevents quality degradation over time:

```python
@pytest.mark.asyncio
async def test_regression_prevention_suite():
    baseline = QualityBaseline()
    results = []
    
    for text, voice, complexity in regression_cases:
        metrics = await analyzer.analyze_comprehensive_quality(
            text, voice, inference_engine, audio_processor
        )
        results.append(metrics)
    
    # Validate against baseline thresholds
    assert all(m.transcription_accuracy >= baseline.min_transcription_accuracy 
              for m in results)
```

### 3. Text Complexity Assessment

Automatically categorizes text difficulty:

```python
def assess_text_complexity(text):
    word_count = len(text.split())
    avg_word_length = mean([len(word) for word in text.split()])
    sentence_count = len([s for s in text.split('.') if s.strip()])
    
    # Returns: 'simple', 'medium', 'complex'
```

## Usage Examples

### 1. Single Sample Analysis

```python
from jabbertts.scripts.simple_perceptual_test import analyze_simple_perceptual_quality

metrics = await analyze_simple_perceptual_quality(
    text="The weather today is beautiful.",
    voice="fable"
)

print(f"Human Likeness: {metrics.human_likeness:.1f}%")
print(f"Prosody Score: {metrics.prosody_score:.1f}%")
print(f"Transcription Accuracy: {metrics.transcription_accuracy:.1f}%")
```

### 2. Comprehensive Test Suite

```python
import pytest

# Run full perceptual quality test suite
pytest.main(["tests/test_perceptual_quality.py", "-v"])
```

### 3. Regression Monitoring

```python
# Automated regression testing
results = await test_regression_prevention_suite()

if results["failures"]:
    print("⚠️ Quality regression detected!")
    for failure in results["failures"]:
        print(f"  - {failure}")
else:
    print("✅ No quality regression detected")
```

## Quality Assessment Scoring

### Intelligibility Status
- **🟢 EXCELLENT**: Accuracy ≥95%
- **🟡 GOOD**: Accuracy ≥80%  
- **🟠 POOR**: Accuracy ≥50%
- **🔴 UNINTELLIGIBLE**: Accuracy <50%

### Human-Likeness Status
- **🟢 VERY HUMAN-LIKE**: Score ≥85%
- **🟡 MODERATELY HUMAN-LIKE**: Score ≥70%
- **🟠 SOMEWHAT ROBOTIC**: Score ≥50%
- **🔴 VERY ROBOTIC**: Score <50%

### Prosody Quality Status
- **🟢 NATURAL PROSODY**: Score ≥80%
- **🟡 ACCEPTABLE PROSODY**: Score ≥60%
- **🔴 POOR PROSODY**: Score <60%

## Integration with Dashboard

The perceptual quality metrics are integrated into the dashboard debug interface:

1. **Real-time Analysis**: Live perceptual quality scoring
2. **Visual Indicators**: Color-coded quality status
3. **Detailed Breakdown**: Individual metric scores
4. **Historical Tracking**: Quality trends over time

## Configuration and Customization

### 1. Adjust Quality Thresholds

```python
# Custom baseline for stricter quality requirements
strict_baseline = QualityBaseline(
    min_transcription_accuracy=98.0,
    min_human_likeness=85.0,
    min_prosody=80.0
)
```

### 2. Weight Adjustment

```python
# Customize human-likeness calculation weights
custom_weights = {
    'naturalness_score': 0.4,    # Increase naturalness importance
    'prosody_score': 0.3,        # Increase prosody importance  
    'clarity_score': 0.2,
    'emotional_expression': 0.1,
    'rhythm_score': 0.0          # Disable rhythm scoring
}
```

### 3. Test Case Customization

```python
# Add domain-specific test cases
MEDICAL_TEST_CASES = [
    ("Take two tablets twice daily.", "clinical", "medium"),
    ("The patient shows signs of improvement.", "professional", "complex"),
]
```

## Best Practices

### 1. Regular Quality Monitoring
- Run perceptual quality tests daily
- Monitor trends and regression patterns
- Set up automated alerts for quality drops

### 2. Comprehensive Testing
- Test across all voice models
- Include diverse text complexities
- Validate edge cases and corner scenarios

### 3. Baseline Management
- Update baselines as system improves
- Document threshold changes
- Maintain historical quality records

### 4. Performance Balance
- Balance quality vs performance requirements
- Optimize for target use cases
- Consider real-time constraints

## Troubleshooting

### Common Issues

#### 1. Low Human-Likeness Scores
- **Check prosody analysis**: Verify speech rate and rhythm
- **Review emotional expression**: Ensure spectral variation
- **Validate naturalness**: Check technical quality metrics

#### 2. Inconsistent Results
- **Audio quality**: Ensure consistent input quality
- **Model warmup**: Allow proper model initialization
- **Sample rate**: Verify consistent audio processing

#### 3. Performance Issues
- **Model size**: Consider smaller Whisper models for speed
- **Batch processing**: Process multiple samples efficiently
- **Caching**: Cache model loading and initialization

## API Reference

### PerceptualQualityMetrics
```python
@dataclass
class PerceptualQualityMetrics:
    transcription_accuracy: float
    word_error_rate: float
    character_error_rate: float
    overall_quality: float
    naturalness_score: float
    clarity_score: float
    consistency_score: float
    prosody_score: float
    rhythm_score: float
    emotional_expression: float
    human_likeness: float
    rtf: float
    inference_time: float
    audio_duration: float
    voice: str
    text_complexity: str
    timestamp: str
```

### PerceptualQualityAnalyzer Methods
```python
async def analyze_comprehensive_quality(text, voice, engine, processor) -> PerceptualQualityMetrics
def analyze_prosody_and_rhythm(audio_data, sample_rate, text) -> Tuple[float, float]
def analyze_emotional_expression(audio_data, sample_rate, text) -> float
def calculate_human_likeness(metrics) -> float
def assess_text_complexity(text) -> str
```
