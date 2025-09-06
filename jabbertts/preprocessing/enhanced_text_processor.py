"""Enhanced Text Preprocessing with Optimized eSpeak-NG Integration.

This module provides comprehensive text preprocessing with intelligent phonemization,
improved punctuation handling, and model-specific optimization for JabberTTS.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import unicodedata

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """TTS model types with different preprocessing requirements."""
    SPEECHT5 = "speecht5"
    OPENAUDIO = "openaudio"
    COQUI_VITS = "coqui_vits"
    UNKNOWN = "unknown"


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    enable_phonemization: bool = True
    enable_text_normalization: bool = True
    enable_punctuation_enhancement: bool = True
    enable_prosody_markers: bool = True
    preserve_case: bool = False
    expand_abbreviations: bool = True
    normalize_numbers: bool = True
    language: str = "en"
    model_type: ModelType = ModelType.UNKNOWN


class EnhancedTextProcessor:
    """Enhanced text processor with optimized eSpeak-NG integration."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize enhanced text processor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.phonemizer = None
        self.phonemizer_backend = None
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Model-specific phonemization settings
        self.model_phonemization_map = {
            ModelType.SPEECHT5: False,      # SpeechT5 works better without phonemization
            ModelType.OPENAUDIO: True,      # OpenAudio benefits from phonemization
            ModelType.COQUI_VITS: True,     # Coqui VITS can use phonemization
            ModelType.UNKNOWN: False        # Default to safe option
        }
        
        # Initialize components
        self._init_phonemizer()
        self._init_text_normalizers()
        
        logger.info(f"Enhanced text processor initialized for {self.config.model_type.value}")
    
    def _init_phonemizer(self) -> None:
        """Initialize phonemizer with enhanced error handling."""
        if not self.config.enable_phonemization:
            logger.info("Phonemization disabled by configuration")
            return
        
        try:
            from phonemizer import phonemize
            from phonemizer.backend import EspeakBackend
            
            # Test basic functionality
            test_result = phonemize(
                "Hello world",
                language='en-us',
                backend='espeak',
                strip=True,
                preserve_punctuation=True,
                with_stress=True
            )
            
            self.phonemizer = phonemize
            
            # Create optimized backend with enhanced configuration
            try:
                self.phonemizer_backend = EspeakBackend(
                    language='en-us',
                    preserve_punctuation=True,
                    with_stress=True,
                    tie=False,
                    language_switch='remove-flags',
                    words_mismatch='ignore'  # Handle word mismatches gracefully
                )
                logger.info("Optimized eSpeak backend created successfully")
                
                # Test backend functionality
                backend_test = self.phonemizer_backend.phonemize(["Hello world"], strip=True)
                logger.info(f"Backend test successful: {backend_test}")
                
            except Exception as backend_error:
                logger.warning(f"Could not create optimized backend: {backend_error}")
                self.phonemizer_backend = None
            
            logger.info(f"Phonemizer initialized successfully. Test: '{test_result}'")
            
        except ImportError as e:
            logger.warning(f"Phonemizer not available: {e}")
            logger.info("Install with: pip install phonemizer")
            self.config.enable_phonemization = False
            
        except Exception as e:
            logger.error(f"Failed to initialize phonemizer: {e}")
            self.config.enable_phonemization = False
    
    def _init_text_normalizers(self) -> None:
        """Initialize text normalization components."""
        # Number normalization
        try:
            import inflect
            self.inflect_engine = inflect.engine()
            logger.info("inflect engine initialized for number normalization")
        except ImportError:
            logger.warning("inflect not available, number normalization limited")
            self.inflect_engine = None
        
        # Abbreviation dictionary
        self.abbreviations = {
            "mr.": "mister",
            "mrs.": "missus", 
            "ms.": "miss",
            "dr.": "doctor",
            "prof.": "professor",
            "st.": "street",
            "ave.": "avenue",
            "blvd.": "boulevard",
            "rd.": "road",
            "etc.": "etcetera",
            "vs.": "versus",
            "e.g.": "for example",
            "i.e.": "that is",
            "a.m.": "ay em",
            "p.m.": "pee em",
            "u.s.": "united states",
            "u.k.": "united kingdom"
        }
        
        # Punctuation enhancement patterns
        self.punctuation_patterns = {
            # Multiple punctuation marks
            r'[.]{2,}': '...',  # Normalize ellipsis
            r'[!]{2,}': '!',    # Multiple exclamations
            r'[?]{2,}': '?',    # Multiple questions
            
            # Spacing around punctuation
            r'\s*([,.!?;:])\s*': r'\1 ',  # Normalize spacing
            r'\s+': ' ',  # Multiple spaces to single space
        }
    
    def process_text(
        self,
        text: str,
        model_type: Optional[ModelType] = None,
        language: Optional[str] = None
    ) -> str:
        """Process text with model-specific optimizations.
        
        Args:
            text: Input text to process
            model_type: Target model type for optimization
            language: Language code for processing
            
        Returns:
            Processed text optimized for the target model
        """
        if not text or not text.strip():
            return ""
        
        # Update configuration if provided
        if model_type:
            self.config.model_type = model_type
        if language:
            self.config.language = language
        
        # Determine if phonemization should be used for this model
        should_phonemize = self._should_use_phonemization(self.config.model_type)
        
        logger.debug(f"Processing text for {self.config.model_type.value}, phonemization: {should_phonemize}")
        
        # Processing pipeline
        processed_text = text
        
        # 1. Basic cleaning and normalization
        processed_text = self._basic_cleaning(processed_text)
        processed_text = self._normalize_unicode(processed_text)
        
        # 2. Text normalization
        if self.config.enable_text_normalization:
            processed_text = self._normalize_text(processed_text)
        
        # 3. Punctuation enhancement
        if self.config.enable_punctuation_enhancement:
            processed_text = self._enhance_punctuation(processed_text)
        
        # 4. Prosody markers (if enabled)
        if self.config.enable_prosody_markers:
            processed_text = self._add_prosody_markers(processed_text)
        
        # 5. Phonemization (model-specific)
        if should_phonemize and self.phonemizer:
            try:
                processed_text = self._phonemize_text(processed_text, self.config.language)
            except Exception as e:
                logger.warning(f"Phonemization failed, using original text: {e}")
        
        # 6. Final cleanup
        processed_text = self._final_cleanup(processed_text)
        
        logger.debug(f"Text processing complete: '{text[:50]}...' -> '{processed_text[:50]}...'")
        
        return processed_text
    
    def _should_use_phonemization(self, model_type: ModelType) -> bool:
        """Determine if phonemization should be used for the given model type."""
        if not self.config.enable_phonemization:
            return False
        
        return self.model_phonemization_map.get(model_type, False)
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common Unicode punctuation with ASCII equivalents
        replacements = {
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            ''': "'",  # Left single quotation mark
            ''': "'",  # Right single quotation mark
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '…': '...',  # Horizontal ellipsis
        }
        
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Comprehensive text normalization."""
        # Expand abbreviations
        if self.config.expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        # Normalize numbers
        if self.config.normalize_numbers:
            text = self._normalize_numbers(text)
        
        # Handle special characters and symbols
        text = self._normalize_symbols(text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbrev, expansion in self.abbreviations.items():
            # Case-insensitive replacement with word boundaries
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Normalize numbers to written form."""
        if not self.inflect_engine:
            return text
        
        def number_to_words(match):
            number_str = match.group()
            try:
                # Handle different number formats
                if '.' in number_str:
                    # Decimal number
                    parts = number_str.split('.')
                    integer_part = self.inflect_engine.number_to_words(int(parts[0]))
                    decimal_part = ' '.join(self.inflect_engine.number_to_words(int(d)) for d in parts[1])
                    return f"{integer_part} point {decimal_part}"
                else:
                    # Integer
                    return self.inflect_engine.number_to_words(int(number_str))
            except (ValueError, OverflowError):
                return number_str
        
        # Replace standalone numbers
        text = re.sub(r'\b\d+(?:\.\d+)?\b', number_to_words, text)
        
        return text
    
    def _normalize_symbols(self, text: str) -> str:
        """Normalize symbols and special characters."""
        symbol_replacements = {
            '&': ' and ',
            '@': ' at ',
            '#': ' hash ',
            '%': ' percent ',
            '$': ' dollar ',
            '€': ' euro ',
            '£': ' pound ',
            '+': ' plus ',
            '=': ' equals ',
            '<': ' less than ',
            '>': ' greater than ',
        }
        
        for symbol, replacement in symbol_replacements.items():
            text = text.replace(symbol, replacement)
        
        return text
    
    def _enhance_punctuation(self, text: str) -> str:
        """Enhance punctuation for better prosody."""
        # Apply punctuation patterns
        for pattern, replacement in self.punctuation_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Add pauses for better prosody
        text = re.sub(r'([.!?])\s+', r'\1 ', text)  # Ensure space after sentence endings
        text = re.sub(r'([,;:])\s*', r'\1 ', text)  # Ensure space after clause separators
        
        return text
    
    def _add_prosody_markers(self, text: str) -> str:
        """Add prosody markers for better speech synthesis."""
        # Emphasis for ALL CAPS words
        text = re.sub(r'\b[A-Z]{2,}\b', r'<emphasis>\g<0></emphasis>', text)
        
        # Question intonation
        text = re.sub(r'([^.!?]*\?)', r'<prosody pitch="+10%">\1</prosody>', text)
        
        # Exclamation emphasis
        text = re.sub(r'([^.!?]*!)', r'<prosody volume="+20%">\1</prosody>', text)
        
        return text
    
    def _phonemize_text(self, text: str, language: str) -> str:
        """Phonemize text with enhanced error handling."""
        if not self.phonemizer:
            return text
        
        # Check cache
        cache_key = f"{text}:{language}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        try:
            # Map language codes
            lang_map = {
                'en': 'en-us', 'es': 'es', 'fr': 'fr-fr', 'de': 'de',
                'it': 'it', 'pt': 'pt', 'ru': 'ru', 'zh': 'zh',
                'ja': 'ja', 'ko': 'ko', 'ar': 'ar'
            }
            
            espeak_lang = lang_map.get(language, 'en-us')
            
            # Use backend if available, otherwise use direct phonemization
            if self.phonemizer_backend:
                phonemes = self.phonemizer_backend.phonemize([text], strip=True)[0]
            else:
                phonemes = self.phonemizer(
                    text,
                    language=espeak_lang,
                    backend='espeak',
                    strip=True,
                    preserve_punctuation=True,
                    with_stress=True
                )
            
            # Post-process phonemes
            phonemes = self._post_process_phonemes(phonemes)
            
            # Cache result
            self.cache[cache_key] = phonemes
            
            # Limit cache size
            if len(self.cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.cache.keys())[:100]
                for key in oldest_keys:
                    del self.cache[key]
            
            return phonemes
            
        except Exception as e:
            logger.warning(f"Phonemization failed for '{text[:50]}...': {e}")
            return text
    
    def _post_process_phonemes(self, phonemes: str) -> str:
        """Post-process phonemes for better quality."""
        # Clean up common phonemization artifacts
        phonemes = re.sub(r'\s+', ' ', phonemes)  # Normalize spaces
        phonemes = re.sub(r'([.!?])\s*', r'\1 ', phonemes)  # Ensure space after punctuation
        
        return phonemes.strip()
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup."""
        # Remove excessive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence spacing
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Strip and return
        return text.strip()
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get processing statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "phonemization_enabled": self.config.enable_phonemization and self.phonemizer is not None
        }


# Global instance
_text_processor = None


def get_enhanced_text_processor(config: Optional[PreprocessingConfig] = None) -> EnhancedTextProcessor:
    """Get the global enhanced text processor instance."""
    global _text_processor
    if _text_processor is None or (config and config != _text_processor.config):
        _text_processor = EnhancedTextProcessor(config)
    return _text_processor
