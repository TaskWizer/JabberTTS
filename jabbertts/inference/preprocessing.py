"""Text Preprocessing for TTS.

This module handles text normalization, cleaning, and preprocessing
before feeding text to TTS models.
"""

import re
import logging
from typing import Optional, Dict, Any
import unicodedata

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessor for TTS input.
    
    Handles text normalization, cleaning, and preparation for TTS models.
    """
    
    def __init__(self, use_phonemizer: bool = True):
        """Initialize text preprocessor.
        
        Args:
            use_phonemizer: Whether to use phonemizer for phonetic conversion
        """
        self.use_phonemizer = use_phonemizer
        self.phonemizer = None
        
        if use_phonemizer:
            self._init_phonemizer()
    
    def _init_phonemizer(self) -> None:
        """Initialize phonemizer if available."""
        try:
            from phonemizer import phonemize
            from phonemizer.backend import EspeakBackend
            
            # Test if espeak-ng is available
            backend = EspeakBackend('en-us')
            self.phonemizer = phonemize
            logger.info("Phonemizer initialized successfully")
            
        except ImportError:
            logger.warning("Phonemizer not available, install with: pip install phonemizer")
            self.use_phonemizer = False
        except Exception as e:
            logger.warning(f"Could not initialize phonemizer: {e}")
            self.use_phonemizer = False
    
    def preprocess(self, text: str, language: str = "en") -> str:
        """Preprocess text for TTS.
        
        Args:
            text: Input text to preprocess
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Preprocessed text ready for TTS
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Basic cleaning and normalization
        processed_text = self._basic_cleaning(text)
        processed_text = self._normalize_unicode(processed_text)
        processed_text = self._expand_abbreviations(processed_text)
        processed_text = self._normalize_numbers(processed_text)
        processed_text = self._handle_punctuation(processed_text)
        
        # Apply phonemization if available and requested
        if self.use_phonemizer and self.phonemizer:
            try:
                processed_text = self._phonemize(processed_text, language)
            except Exception as e:
                logger.warning(f"Phonemization failed, using original text: {e}")
        
        return processed_text.strip()
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove or replace problematic characters
        text = text.replace('\t', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common Unicode quotes and dashes
        replacements = {
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            ''': "'",  # Left single quotation mark
            ''': "'",  # Right single quotation mark
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '…': '...',  # Horizontal ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded abbreviations
        """
        # Common abbreviations
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            r'\bSt\.': 'Saint',
            r'\bAve\.': 'Avenue',
            r'\bBlvd\.': 'Boulevard',
            r'\bRd\.': 'Road',
            r'\bSt\.': 'Street',
            r'\betc\.': 'et cetera',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
            r'\bvs\.': 'versus',
        }
        
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Normalize numbers to written form.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized numbers
        """
        try:
            import inflect
            p = inflect.engine()
            
            # Convert numbers to words
            def number_to_words(match):
                number = match.group()
                try:
                    return p.number_to_words(number)
                except:
                    return number
            
            # Replace standalone numbers
            text = re.sub(r'\b\d+\b', number_to_words, text)
            
        except ImportError:
            logger.debug("inflect not available for number normalization")
        except Exception as e:
            logger.warning(f"Number normalization failed: {e}")
        
        return text
    
    def _handle_punctuation(self, text: str) -> str:
        """Handle punctuation for better TTS.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized punctuation
        """
        # Add pauses for better speech rhythm
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        text = re.sub(r'([,;:])\s*', r'\1 ', text)
        
        # Ensure sentences end with proper punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def _phonemize(self, text: str, language: str) -> str:
        """Convert text to phonemes using phonemizer.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Phonemized text
        """
        if not self.phonemizer:
            return text
        
        try:
            # Map language codes to espeak language codes
            lang_map = {
                'en': 'en-us',
                'es': 'es',
                'fr': 'fr-fr',
                'de': 'de',
                'it': 'it',
                'pt': 'pt',
                'ru': 'ru',
                'zh': 'zh',
                'ja': 'ja',
                'ko': 'ko',
                'ar': 'ar'
            }
            
            espeak_lang = lang_map.get(language, 'en-us')
            
            # Phonemize the text
            phonemes = self.phonemizer(
                text,
                language=espeak_lang,
                backend='espeak',
                strip=True,
                preserve_punctuation=True,
                with_stress=True
            )
            
            return phonemes
            
        except Exception as e:
            logger.warning(f"Phonemization failed for language {language}: {e}")
            return text
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about preprocessing capabilities.
        
        Returns:
            Dictionary with preprocessing information
        """
        return {
            "phonemizer_available": self.use_phonemizer,
            "supported_features": [
                "unicode_normalization",
                "abbreviation_expansion", 
                "number_normalization",
                "punctuation_handling",
                "basic_cleaning"
            ] + (["phonemization"] if self.use_phonemizer else [])
        }
