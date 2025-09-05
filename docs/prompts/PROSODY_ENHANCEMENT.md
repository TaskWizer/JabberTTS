# Prosodic Feature Optimization Prompt

---
**📚 LiteTTS Documentation Navigation**

**Core Documentation:** [Features](../FEATURES.md) | [Configuration](../CONFIGURATION.md) | [Performance](../PERFORMANCE.md) | [Monitoring](../MONITORING.md) | [Testing](../TESTING.md) | [Troubleshooting](../TROUBLESHOOTING.md)

**Setup & Usage:** [Dependencies](../DEPENDENCIES.md) | [Quick Start](../usage/QUICK_START_COMMANDS.md) | [Docker Deployment](../usage/DOCKER-DEPLOYMENT.md) | [OpenWebUI Integration](../usage/OPENWEBUI-INTEGRATION.md)

**Advanced:** [API Reference](../api/API_REFERENCE.md) | [Development](../development/README.md) | [Voice System](../voices/README.md) | [Watermarking](../WATERMARKING.md)

**Project:** [Changelog](../CHANGELOG.md) | [Roadmap](../ROADMAP.md) | [Contributing](../CONTRIBUTIONS.md) | [Beta Features](../BETA_FEATURES.md)

---

## Objective
Optimize prosodic features in TTS synthesis to achieve human-like speech patterns with natural stress, intonation, and rhythm that enhance intelligibility and expressiveness.

## Core Prosodic Principles

### 1. Stress Pattern Optimization
**Primary Focus Areas:**
- **Lexical Stress**: Correct syllable emphasis within words
- **Sentence Stress**: Appropriate emphasis on content words
- **Contrastive Stress**: Emphasis for meaning disambiguation
- **Emphatic Stress**: Emotional or pragmatic emphasis

**Implementation Guidelines:**
```
When processing text for stress assignment:
1. Identify content words (nouns, verbs, adjectives, adverbs) for primary stress
2. Reduce stress on function words (articles, prepositions, auxiliaries)
3. Apply contrastive stress for new or contrasted information
4. Consider emotional context for emphatic stress placement
5. Maintain natural stress hierarchy within phrases
```

### 2. Intonation Curve Generation
**Intonation Pattern Types:**
- **Declarative**: Gradual fall from high to low pitch
- **Yes/No Questions**: Rising intonation at the end
- **Wh-Questions**: Falling intonation after question word
- **Lists**: Rising on non-final items, falling on final item
- **Continuation**: Level or slight rise to indicate more to come

**Context-Sensitive Intonation:**
```
Adapt intonation patterns based on:
- Sentence type and syntactic structure
- Emotional state and speaker attitude
- Discourse context and conversation flow
- Cultural and regional variation patterns
- Speaker personality and style preferences
```

### 3. Rhythm and Timing Optimization
**Temporal Feature Control:**
- **Syllable Duration**: Variable based on stress and position
- **Pause Placement**: Syntactic and semantic boundary marking
- **Speech Rate**: Context-appropriate speed variation
- **Rhythm Type**: Stress-timed vs. syllable-timed patterns

**Natural Timing Principles:**
```
Apply timing rules that reflect natural speech:
1. Stressed syllables are longer than unstressed syllables
2. Final syllables in phrases are lengthened (final lengthening)
3. Pauses occur at major syntactic boundaries
4. Speech rate varies with content complexity and emotional state
5. Rhythm maintains perceptual regularity while allowing natural variation
```

## Advanced Prosodic Techniques

### 1. Hierarchical Prosodic Structure
**Prosodic Hierarchy Levels:**
- **Syllable**: Basic rhythmic unit
- **Foot**: Stress-based grouping
- **Prosodic Word**: Lexical stress domain
- **Prosodic Phrase**: Intermediate grouping
- **Intonational Phrase**: Complete intonational unit

**Implementation Strategy:**
```
Build prosodic structure bottom-up:
1. Assign syllable-level features (duration, pitch, amplitude)
2. Group syllables into feet based on stress patterns
3. Organize feet into prosodic words with primary stress
4. Form prosodic phrases at syntactic boundaries
5. Create intonational phrases with coherent pitch contours
```

### 2. Contextual Prosodic Adaptation
**Discourse-Level Prosody:**
- **Topic Introduction**: Higher pitch, slower rate, clear articulation
- **Topic Development**: Moderate prosody with appropriate emphasis
- **Topic Conclusion**: Lower pitch, final lengthening, closure signals
- **Topic Shift**: Prosodic reset with new baseline establishment

**Emotional Prosodic Modulation:**
```
Adjust prosodic parameters for emotional expression:
- Joy: Higher pitch, wider range, faster rate, bright timbre
- Sadness: Lower pitch, narrow range, slower rate, breathy voice
- Anger: Sharp pitch changes, increased intensity, irregular rhythm
- Fear: Higher pitch, tremulous patterns, faster rate, tense voice
- Surprise: Sudden pitch rise, increased range, abrupt timing changes
```

### 3. Speaker-Specific Prosodic Modeling
**Individual Prosodic Characteristics:**
- **Pitch Range**: Personal fundamental frequency boundaries
- **Speech Rate**: Individual tempo preferences and capabilities
- **Rhythm Patterns**: Personal timing and stress characteristics
- **Intonation Style**: Individual melodic preferences and habits

## Quality Optimization Guidelines

### 1. Prosodic Coherence
**Coherence Principles:**
```
Ensure prosodic consistency and naturalness:
1. Maintain pitch range consistency within speakers
2. Apply gradual transitions between prosodic states
3. Respect physiological constraints on pitch and timing
4. Avoid abrupt or unnatural prosodic changes
5. Ensure prosodic patterns support semantic content
```

### 2. Perceptual Optimization
**Perceptual Priorities:**
- **Intelligibility**: Prosody should enhance, not hinder, understanding
- **Naturalness**: Patterns should sound authentically human
- **Expressiveness**: Appropriate emotional and pragmatic communication
- **Comfort**: Suitable for extended listening without fatigue

**Optimization Strategies:**
```
Optimize for human perception:
1. Prioritize prosodic cues that enhance intelligibility
2. Use just-noticeable differences (JND) for parameter adjustment
3. Apply perceptual weighting to different prosodic features
4. Consider masking effects and perceptual salience
5. Validate with subjective listening tests
```

### 3. Context-Aware Quality Control
**Quality Metrics by Context:**
- **Conversational**: Natural dialogue rhythm, appropriate turn-taking cues
- **Narrative**: Engaging storytelling prosody, character differentiation
- **Instructional**: Clear emphasis on key information, appropriate pacing
- **Formal**: Professional intonation patterns, authoritative delivery

## Implementation Best Practices

### 1. Prosodic Feature Extraction
**Text Analysis Pipeline:**
```
Extract prosodic information from text:
1. Syntactic parsing for phrase boundary identification
2. Semantic analysis for content word emphasis
3. Pragmatic analysis for discourse-level prosody
4. Emotional analysis for affective prosodic modulation
5. Context analysis for speaker and situation adaptation
```

### 2. Prosodic Parameter Generation
**Parameter Calculation:**
```
Generate prosodic parameters systematically:
1. Calculate base prosodic values from linguistic analysis
2. Apply contextual modifications for discourse and emotion
3. Add speaker-specific characteristics and preferences
4. Introduce natural variation and micro-prosodic details
5. Validate parameters against perceptual and physiological constraints
```

### 3. Real-Time Prosodic Processing
**Efficiency Considerations:**
```
Optimize prosodic processing for real-time applications:
1. Precompute common prosodic patterns and templates
2. Use efficient algorithms for prosodic parameter calculation
3. Implement caching strategies for repeated prosodic structures
4. Apply progressive enhancement based on available processing time
5. Maintain quality graceful degradation under time constraints
```

## Evaluation and Validation

### 1. Objective Prosodic Metrics
**Acoustic Measurements:**
- **F0 Contour Accuracy**: Comparison with target intonation patterns
- **Duration Pattern Naturalness**: Timing appropriateness assessment
- **Stress Placement Correctness**: Accuracy of stress assignment
- **Prosodic Boundary Detection**: Phrase boundary marking effectiveness

### 2. Subjective Prosodic Evaluation
**Perceptual Studies:**
- **Naturalness Ratings**: Human judgments of prosodic authenticity
- **Preference Tests**: Comparative evaluation against alternatives
- **Intelligibility Assessment**: Understanding enhancement measurement
- **Expressiveness Evaluation**: Emotional and pragmatic communication effectiveness

### 3. Context-Specific Validation
**Application-Specific Testing:**
```
Validate prosodic quality in target applications:
1. Conversational AI: Natural dialogue flow and turn-taking
2. Audiobook narration: Engaging storytelling and character voices
3. Educational content: Clear instruction delivery and emphasis
4. Customer service: Professional and empathetic communication
5. Entertainment: Expressive and engaging performance
```

## Configuration and Customization

### 1. Prosodic Quality Levels
**Processing Modes:**
- **Fast**: Basic stress and intonation patterns, minimal context consideration
- **Balanced**: Moderate prosodic complexity with good performance
- **High-Quality**: Full prosodic modeling with advanced context analysis

### 2. User Customization Options
**Adjustable Parameters:**
- **Prosodic Emphasis Strength**: Control over stress and intonation prominence
- **Intonation Range**: Adjustment of pitch variation extent
- **Speech Rate Preference**: User-specific tempo settings
- **Emotional Expressiveness**: Control over affective prosodic modulation

### 3. Context-Specific Presets
**Predefined Configurations:**
- **Conversational**: Optimized for natural dialogue
- **Professional**: Formal presentation and business communication
- **Educational**: Clear instruction and information delivery
- **Entertainment**: Expressive and engaging performance

---

*This prompt guides the optimization of prosodic features to achieve natural, expressive, and contextually appropriate speech synthesis in the Kokoro TTS system.*
