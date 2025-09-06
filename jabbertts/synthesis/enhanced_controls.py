"""Enhanced Control Parameters for JabberTTS.

This module implements advanced synthesis controls including emotion, prosody,
speaking styles, and multi-language support while maintaining OpenAI API compatibility.
"""

import logging
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Supported emotion types."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    SURPRISED = "surprised"
    FEARFUL = "fearful"


class SpeakingStyle(Enum):
    """Supported speaking styles."""
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    DRAMATIC = "dramatic"
    WHISPERED = "whispered"
    SHOUTING = "shouting"
    STORYTELLING = "storytelling"
    NEWS_ANCHOR = "news_anchor"
    POETRY = "poetry"


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


@dataclass
class EmotionControl:
    """Emotion control parameters."""
    emotion_type: EmotionType = EmotionType.NEUTRAL
    intensity: float = 0.5  # 0.0 to 1.0
    blend_emotions: Dict[EmotionType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate emotion parameters."""
        self.intensity = max(0.0, min(1.0, self.intensity))
        
        # Normalize blend emotions
        if self.blend_emotions:
            total_weight = sum(self.blend_emotions.values())
            if total_weight > 0:
                self.blend_emotions = {
                    emotion: weight / total_weight
                    for emotion, weight in self.blend_emotions.items()
                }


@dataclass
class ProsodyControl:
    """Prosody control parameters."""
    pitch_shift: float = 0.0  # -50% to +50% (semitones)
    speed: float = 1.0  # 0.25x to 4.0x
    volume: float = 1.0  # 0.0 to 2.0
    emphasis_words: List[str] = field(default_factory=list)
    pause_after_words: Dict[str, float] = field(default_factory=dict)  # word -> pause_seconds
    
    def __post_init__(self):
        """Validate prosody parameters."""
        self.pitch_shift = max(-50.0, min(50.0, self.pitch_shift))
        self.speed = max(0.25, min(4.0, self.speed))
        self.volume = max(0.0, min(2.0, self.volume))


@dataclass
class SpeechStyle:
    """Speaking style parameters."""
    style: SpeakingStyle = SpeakingStyle.CONVERSATIONAL
    formality: float = 0.5  # 0.0 (very casual) to 1.0 (very formal)
    energy: float = 0.5  # 0.0 (low energy) to 1.0 (high energy)
    articulation: float = 0.5  # 0.0 (relaxed) to 1.0 (precise)
    
    def __post_init__(self):
        """Validate style parameters."""
        self.formality = max(0.0, min(1.0, self.formality))
        self.energy = max(0.0, min(1.0, self.energy))
        self.articulation = max(0.0, min(1.0, self.articulation))


@dataclass
class LanguageControl:
    """Language and accent control."""
    language: Language = Language.ENGLISH
    accent: Optional[str] = None  # e.g., "british", "american", "australian"
    auto_detect: bool = True
    pronunciation_guide: Dict[str, str] = field(default_factory=dict)  # word -> phonetic
    
    def get_language_code(self) -> str:
        """Get the language code."""
        return self.language.value


@dataclass
class EnhancedSynthesisParams:
    """Complete enhanced synthesis parameters."""
    # Core parameters (OpenAI compatible)
    text: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0
    
    # Enhanced parameters
    emotion: Optional[EmotionControl] = None
    prosody: Optional[ProsodyControl] = None
    style: Optional[SpeechStyle] = None
    language: Optional[LanguageControl] = None
    
    # Advanced features
    enable_ssml: bool = False
    custom_voice_id: Optional[str] = None  # For voice cloning
    streaming: bool = False
    
    def __post_init__(self):
        """Initialize default enhanced parameters."""
        if self.emotion is None:
            self.emotion = EmotionControl()
        if self.prosody is None:
            self.prosody = ProsodyControl(speed=self.speed)
        if self.style is None:
            self.style = SpeechStyle()
        if self.language is None:
            self.language = LanguageControl()


class SSMLProcessor:
    """Process SSML (Speech Synthesis Markup Language) tags."""
    
    def __init__(self):
        """Initialize SSML processor."""
        self.ssml_patterns = {
            'break': re.compile(r'<break\s+time="([^"]+)"\s*/?>'),
            'emphasis': re.compile(r'<emphasis\s+level="([^"]+)">([^<]+)</emphasis>'),
            'prosody': re.compile(r'<prosody\s+([^>]+)>([^<]+)</prosody>'),
            'say_as': re.compile(r'<say-as\s+interpret-as="([^"]+)">([^<]+)</say-as>'),
            'phoneme': re.compile(r'<phoneme\s+ph="([^"]+)">([^<]+)</phoneme>')
        }
    
    def process_ssml(self, text: str, params: EnhancedSynthesisParams) -> Tuple[str, EnhancedSynthesisParams]:
        """Process SSML tags and extract synthesis parameters.
        
        Args:
            text: Text potentially containing SSML tags
            params: Current synthesis parameters
            
        Returns:
            Tuple of (cleaned_text, updated_params)
        """
        if not params.enable_ssml:
            return text, params
        
        cleaned_text = text
        updated_params = params
        
        # Process break tags
        breaks = self.ssml_patterns['break'].findall(text)
        for break_time in breaks:
            # Convert break time to pause
            pause_seconds = self._parse_time_value(break_time)
            # Add to prosody control (simplified implementation)
            if updated_params.prosody is None:
                updated_params.prosody = ProsodyControl()
        
        # Process emphasis tags
        emphasis_matches = self.ssml_patterns['emphasis'].finditer(text)
        for match in emphasis_matches:
            level = match.group(1)
            emphasized_text = match.group(2)
            
            # Add to emphasis words
            if updated_params.prosody is None:
                updated_params.prosody = ProsodyControl()
            
            words = emphasized_text.split()
            updated_params.prosody.emphasis_words.extend(words)
        
        # Process prosody tags
        prosody_matches = self.ssml_patterns['prosody'].finditer(text)
        for match in prosody_matches:
            attributes = match.group(1)
            prosody_text = match.group(2)
            
            # Parse prosody attributes
            prosody_params = self._parse_prosody_attributes(attributes)
            
            # Update prosody control
            if updated_params.prosody is None:
                updated_params.prosody = ProsodyControl()
            
            if 'rate' in prosody_params:
                updated_params.prosody.speed = prosody_params['rate']
            if 'pitch' in prosody_params:
                updated_params.prosody.pitch_shift = prosody_params['pitch']
            if 'volume' in prosody_params:
                updated_params.prosody.volume = prosody_params['volume']
        
        # Clean SSML tags from text
        cleaned_text = self._remove_ssml_tags(text)
        
        return cleaned_text, updated_params
    
    def _parse_time_value(self, time_str: str) -> float:
        """Parse time value from SSML (e.g., '500ms', '2s')."""
        if time_str.endswith('ms'):
            return float(time_str[:-2]) / 1000
        elif time_str.endswith('s'):
            return float(time_str[:-1])
        else:
            return float(time_str)
    
    def _parse_prosody_attributes(self, attributes: str) -> Dict[str, float]:
        """Parse prosody attributes from SSML."""
        params = {}
        
        # Simple attribute parsing
        attr_pattern = re.compile(r'(\w+)="([^"]+)"')
        matches = attr_pattern.findall(attributes)
        
        for attr_name, attr_value in matches:
            if attr_name == 'rate':
                if attr_value.endswith('%'):
                    params['rate'] = float(attr_value[:-1]) / 100
                elif attr_value in ['x-slow', 'slow', 'medium', 'fast', 'x-fast']:
                    rate_map = {
                        'x-slow': 0.5, 'slow': 0.75, 'medium': 1.0,
                        'fast': 1.25, 'x-fast': 1.5
                    }
                    params['rate'] = rate_map[attr_value]
                else:
                    params['rate'] = float(attr_value)
            
            elif attr_name == 'pitch':
                if attr_value.endswith('Hz'):
                    # Convert Hz to semitone shift (simplified)
                    hz_value = float(attr_value[:-2])
                    params['pitch'] = (hz_value - 220) / 220 * 12  # Rough conversion
                elif attr_value.endswith('%'):
                    params['pitch'] = (float(attr_value[:-1]) - 100) / 100 * 12
                else:
                    params['pitch'] = float(attr_value)
            
            elif attr_name == 'volume':
                if attr_value.endswith('dB'):
                    db_value = float(attr_value[:-2])
                    params['volume'] = 10 ** (db_value / 20)
                elif attr_value in ['silent', 'x-soft', 'soft', 'medium', 'loud', 'x-loud']:
                    volume_map = {
                        'silent': 0.0, 'x-soft': 0.3, 'soft': 0.6,
                        'medium': 1.0, 'loud': 1.4, 'x-loud': 2.0
                    }
                    params['volume'] = volume_map[attr_value]
                else:
                    params['volume'] = float(attr_value)
        
        return params
    
    def _remove_ssml_tags(self, text: str) -> str:
        """Remove all SSML tags from text."""
        # Remove all XML-like tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text


class LanguageDetector:
    """Automatic language detection for text."""
    
    def __init__(self):
        """Initialize language detector."""
        self.language_patterns = {
            Language.ENGLISH: [
                r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
                r'\b(is|are|was|were|have|has|had|will|would|could|should)\b'
            ],
            Language.SPANISH: [
                r'\b(el|la|los|las|un|una|y|o|pero|en|de|con|por|para)\b',
                r'\b(es|son|está|están|tiene|tienen|será|sería)\b'
            ],
            Language.FRENCH: [
                r'\b(le|la|les|un|une|et|ou|mais|dans|de|avec|par|pour)\b',
                r'\b(est|sont|était|étaient|a|ont|aura|aurait)\b'
            ],
            Language.GERMAN: [
                r'\b(der|die|das|ein|eine|und|oder|aber|in|von|mit|für)\b',
                r'\b(ist|sind|war|waren|hat|haben|wird|würde)\b'
            ]
        }
    
    def detect_language(self, text: str) -> Language:
        """Detect the primary language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language
        """
        text_lower = text.lower()
        language_scores = {}
        
        for language, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            language_scores[language] = score
        
        # Return language with highest score, default to English
        if language_scores:
            detected_language = max(language_scores, key=language_scores.get)
            if language_scores[detected_language] > 0:
                return detected_language
        
        return Language.ENGLISH


class EnhancedControlProcessor:
    """Process enhanced control parameters for synthesis."""
    
    def __init__(self):
        """Initialize the control processor."""
        self.ssml_processor = SSMLProcessor()
        self.language_detector = LanguageDetector()
    
    def process_parameters(self, params: EnhancedSynthesisParams) -> EnhancedSynthesisParams:
        """Process and validate enhanced synthesis parameters.
        
        Args:
            params: Input synthesis parameters
            
        Returns:
            Processed and validated parameters
        """
        # Process SSML if enabled
        if params.enable_ssml:
            params.text, params = self.ssml_processor.process_ssml(params.text, params)
        
        # Auto-detect language if enabled
        if params.language and params.language.auto_detect:
            detected_lang = self.language_detector.detect_language(params.text)
            params.language.language = detected_lang
        
        # Apply emotion-based style adjustments
        if params.emotion and params.style:
            self._apply_emotion_to_style(params.emotion, params.style)
        
        # Apply style-based prosody adjustments
        if params.style and params.prosody:
            self._apply_style_to_prosody(params.style, params.prosody)
        
        return params
    
    def _apply_emotion_to_style(self, emotion: EmotionControl, style: SpeechStyle) -> None:
        """Apply emotion parameters to speaking style."""
        emotion_style_map = {
            EmotionType.HAPPY: {"energy": 0.8, "formality": 0.3},
            EmotionType.SAD: {"energy": 0.2, "formality": 0.4},
            EmotionType.ANGRY: {"energy": 0.9, "formality": 0.6, "articulation": 0.8},
            EmotionType.EXCITED: {"energy": 1.0, "formality": 0.2},
            EmotionType.CALM: {"energy": 0.3, "formality": 0.5, "articulation": 0.7},
            EmotionType.SURPRISED: {"energy": 0.7, "formality": 0.3},
            EmotionType.FEARFUL: {"energy": 0.4, "formality": 0.6}
        }
        
        if emotion.emotion_type in emotion_style_map:
            adjustments = emotion_style_map[emotion.emotion_type]
            intensity = emotion.intensity
            
            for param, target_value in adjustments.items():
                current_value = getattr(style, param)
                # Blend current value with emotion target based on intensity
                new_value = current_value * (1 - intensity) + target_value * intensity
                setattr(style, param, max(0.0, min(1.0, new_value)))
    
    def _apply_style_to_prosody(self, style: SpeechStyle, prosody: ProsodyControl) -> None:
        """Apply speaking style to prosody parameters."""
        style_prosody_map = {
            SpeakingStyle.DRAMATIC: {"pitch_shift": 5.0, "speed": 0.9},
            SpeakingStyle.WHISPERED: {"volume": 0.3, "speed": 0.8},
            SpeakingStyle.SHOUTING: {"volume": 1.5, "pitch_shift": 10.0},
            SpeakingStyle.FORMAL: {"speed": 0.95, "articulation": 0.8},
            SpeakingStyle.CONVERSATIONAL: {"speed": 1.1},
            SpeakingStyle.NEWS_ANCHOR: {"speed": 1.0, "articulation": 0.9},
            SpeakingStyle.POETRY: {"speed": 0.8, "pitch_shift": 3.0}
        }
        
        if style.style in style_prosody_map:
            adjustments = style_prosody_map[style.style]
            
            for param, adjustment in adjustments.items():
                if hasattr(prosody, param):
                    current_value = getattr(prosody, param)
                    if param in ['pitch_shift']:
                        # Additive for pitch
                        new_value = current_value + adjustment * style.energy
                    elif param in ['speed', 'volume']:
                        # Multiplicative for speed and volume
                        new_value = current_value * adjustment
                    else:
                        # Direct assignment for others
                        new_value = adjustment
                    
                    setattr(prosody, param, new_value)
    
    def get_openai_compatible_params(self, params: EnhancedSynthesisParams) -> Dict[str, Any]:
        """Extract OpenAI-compatible parameters.
        
        Args:
            params: Enhanced synthesis parameters
            
        Returns:
            Dictionary of OpenAI-compatible parameters
        """
        openai_params = {
            "input": params.text,
            "voice": params.voice,
            "response_format": params.response_format,
            "speed": params.prosody.speed if params.prosody else params.speed
        }
        
        return openai_params
    
    def get_extended_params(self, params: EnhancedSynthesisParams) -> Dict[str, Any]:
        """Extract extended parameters for advanced models.
        
        Args:
            params: Enhanced synthesis parameters
            
        Returns:
            Dictionary of extended parameters
        """
        extended_params = {}
        
        if params.emotion:
            extended_params["emotion"] = {
                "type": params.emotion.emotion_type.value,
                "intensity": params.emotion.intensity
            }
        
        if params.prosody:
            extended_params["prosody"] = {
                "pitch_shift": params.prosody.pitch_shift,
                "speed": params.prosody.speed,
                "volume": params.prosody.volume
            }
        
        if params.style:
            extended_params["style"] = {
                "speaking_style": params.style.style.value,
                "formality": params.style.formality,
                "energy": params.style.energy
            }
        
        if params.language:
            extended_params["language"] = {
                "code": params.language.get_language_code(),
                "accent": params.language.accent
            }
        
        return extended_params


# Global instance
_control_processor = None


def get_control_processor() -> EnhancedControlProcessor:
    """Get the global enhanced control processor instance."""
    global _control_processor
    if _control_processor is None:
        _control_processor = EnhancedControlProcessor()
    return _control_processor
