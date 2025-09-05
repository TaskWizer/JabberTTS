# Phonemization Stuttering Analysis Report

**Generated**: 2025-09-05 13:06:15
**Analysis Type**: eSpeak-NG Phonemization Impact on Stuttering Artifacts
**Model**: speecht5

## Executive Summary

This analysis specifically tested whether eSpeak-NG phonemization is causing stuttering artifacts by comparing phonemized vs non-phonemized text processing.

## Key Findings

1. CRITICAL: Phonemization has significant impact on stuttering (>0.05 difference)
2. WARNING: Phonemization increases stuttering for 'technical_terms' - score: 0.196 vs 0.132
3. WARNING: High phoneme complexity detected in: technical_terms, stuttering_test

## Recommendations

1. CONTINUE: Phonemization can be safely used - stuttering cause is elsewhere
2. INVESTIGATE: Review phoneme complexity for 'technical_terms' - complexity score: 2.70
3. INVESTIGATE: Review phoneme complexity for 'stuttering_test' - complexity score: 2.80
4. VALIDATE: Generate human listening test samples with phonemization disabled
5. MONITOR: Implement phonemization quality checks in production pipeline
6. DOCUMENT: Update text preprocessing guidelines based on findings
7. TEST: Experiment with alternative phonemization backends if available

## Phonemization Impact Analysis

- **Average RTF Change**: 4.419
- **Average Stuttering Change**: -0.017
- **Performance Improvement**: False
- **Increases Stuttering**: False
- **Stuttering Reduction Cases**: 1
- **Stuttering Increase Cases**: 1

## Test Results Summary

| Test Case | Phonemized RTF | Non-Phonemized RTF | Phonemized Stuttering | Non-Phonemized Stuttering | Difference |
|-----------|----------------|--------------------|-----------------------|----------------------------|------------|
| welcome_word | 21.604 | 0.588 | 0.163 | 0.204 | ➡️ -0.041 |
| tts_phrase | 1.245 | 0.464 | 0.071 | 0.075 | ➡️ -0.004 |
| complex_consonants | 0.628 | 0.427 | 0.063 | 0.097 | ➡️ -0.034 |
| technical_terms | 0.477 | 0.423 | 0.196 | 0.132 | ⬆️ +0.064 |
| stuttering_test | 0.477 | 0.434 | 0.063 | 0.136 | ⬇️ -0.072 |

## Phonemization Pattern Analysis

| Test Case | Original Text | Phonemized Text | Complexity Score | Fragmentation Score |
|-----------|---------------|-----------------|------------------|--------------------|
| welcome_word | Welcome | wˈɛlkʌm. | 0.30 | 1.25 |
| tts_phrase | Text-to-speech synthesis | tˈɛksttəspˈiːtʃ sˈɪnθəsˌɪs. | 1.90 | 1.85 |
| complex_consonants | Strength through struggle | stɹˈɛŋθ θɹuː stɹˈʌɡəl. | 0.90 | 1.36 |
| technical_terms | Neural network architecture optimization | nˈʊɹɹəl nˈɛtwɜːk ˈɑːɹkɪtˌɛktʃɚɹ ˌɑːptɪmᵻzˈeɪʃən. | 2.70 | 1.88 |
| stuttering_test | Testing text-to-speech stuttering artifacts | tˈɛstɪŋ tˈɛksttəspˈiːtʃ stˈʌɾɚɹɪŋ ˈɑːɹɾɪfˌækts. | 2.80 | 1.70 |

## Generated Audio Files

Compare these audio files to identify phonemization impact on stuttering:

### Welcome Word
**Text**: "Welcome"
- Phonemized: `welcome_word_phonemized.wav`
- Non-phonemized: `welcome_word_non_phonemized.wav`

### Tts Phrase
**Text**: "Text-to-speech synthesis"
- Phonemized: `tts_phrase_phonemized.wav`
- Non-phonemized: `tts_phrase_non_phonemized.wav`

### Complex Consonants
**Text**: "Strength through struggle"
- Phonemized: `complex_consonants_phonemized.wav`
- Non-phonemized: `complex_consonants_non_phonemized.wav`

### Technical Terms
**Text**: "Neural network architecture optimization"
- Phonemized: `technical_terms_phonemized.wav`
- Non-phonemized: `technical_terms_non_phonemized.wav`

### Stuttering Test
**Text**: "Testing text-to-speech stuttering artifacts"
- Phonemized: `stuttering_test_phonemized.wav`
- Non-phonemized: `stuttering_test_non_phonemized.wav`

## Detailed Phonemization Analysis

### Welcome Word
**Original**: Welcome
**Phonemized**: wˈɛlkʌm.
**Phoneme Markers**: 1
**Complexity Score**: 0.30
**High Complexity**: False
**Fragmentation Score**: 1.25
**High Fragmentation**: False

### Tts Phrase
**Original**: Text-to-speech synthesis
**Phonemized**: tˈɛksttəspˈiːtʃ sˈɪnθəsˌɪs.
**Phoneme Markers**: 5
**Complexity Score**: 1.90
**High Complexity**: False
**Fragmentation Score**: 1.85
**High Fragmentation**: False

### Complex Consonants
**Original**: Strength through struggle
**Phonemized**: stɹˈɛŋθ θɹuː stɹˈʌɡəl.
**Phoneme Markers**: 3
**Complexity Score**: 0.90
**High Complexity**: False
**Fragmentation Score**: 1.36
**High Fragmentation**: False

### Technical Terms
**Original**: Neural network architecture optimization
**Phonemized**: nˈʊɹɹəl nˈɛtwɜːk ˈɑːɹkɪtˌɛktʃɚɹ ˌɑːptɪmᵻzˈeɪʃən.
**Phoneme Markers**: 9
**Complexity Score**: 2.70
**High Complexity**: True
**Fragmentation Score**: 1.88
**High Fragmentation**: False

### Stuttering Test
**Original**: Testing text-to-speech stuttering artifacts
**Phonemized**: tˈɛstɪŋ tˈɛksttəspˈiːtʃ stˈʌɾɚɹɪŋ ˈɑːɹɾɪfˌækts.
**Phoneme Markers**: 8
**Complexity Score**: 2.80
**High Complexity**: True
**Fragmentation Score**: 1.70
**High Fragmentation**: False

---
**Note**: Manual audio comparison is essential to confirm automated analysis.
**Legend**: ⬆️ = Stuttering increased, ⬇️ = Stuttering decreased, ➡️ = No significant change
