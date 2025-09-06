"""Real-time Streaming TTS Synthesis for JabberTTS.

This module implements context-aware chunking and streaming synthesis with
<500ms first chunk latency and seamless audio concatenation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, AsyncGenerator, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import re
from collections import deque

from jabbertts.models.manager import ModelManager, ModelSelectionCriteria, ModelSelectionStrategy
from jabbertts.audio.processor import get_audio_processor
from jabbertts.inference.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of text chunks for streaming."""
    SENTENCE = "sentence"
    PHRASE = "phrase"
    WORD_GROUP = "word_group"
    EMERGENCY = "emergency"


@dataclass
class TextChunk:
    """A chunk of text for streaming synthesis."""
    text: str
    chunk_type: ChunkType
    priority: int = 0
    context_before: str = ""
    context_after: str = ""
    chunk_id: int = 0
    estimated_duration: float = 0.0


@dataclass
class AudioChunk:
    """A chunk of synthesized audio."""
    audio_data: np.ndarray
    sample_rate: int
    chunk_id: int
    duration: float
    fade_in_samples: int = 0
    fade_out_samples: int = 0
    timestamp: float = 0.0


@dataclass
class StreamingConfig:
    """Configuration for streaming synthesis."""
    min_chunk_size: int = 25
    max_chunk_size: int = 200
    target_chunk_size: int = 100
    max_latency_ms: int = 500
    crossfade_duration_ms: int = 50
    buffer_size: int = 3
    quality_mode: str = "balanced"  # fast, balanced, quality


class ContextAwareChunker:
    """Intelligent text chunking for streaming synthesis."""
    
    def __init__(self, config: StreamingConfig):
        """Initialize the chunker.
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        self.sentence_endings = re.compile(r'[.!?]+\s*')
        self.phrase_breaks = re.compile(r'[,;:]+\s*')
        self.word_boundaries = re.compile(r'\s+')
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text into optimal segments for streaming.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks optimized for streaming
        """
        if len(text) <= self.config.max_chunk_size:
            return [TextChunk(
                text=text.strip(),
                chunk_type=ChunkType.SENTENCE,
                chunk_id=0,
                estimated_duration=self._estimate_duration(text)
            )]
        
        chunks = []
        chunk_id = 0
        
        # First, try to split by sentences
        sentences = self.sentence_endings.split(text)
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add sentence ending back (except for last sentence)
            if i < len(sentences) - 1:
                # Find the original ending
                remaining_text = text[text.find(sentence) + len(sentence):]
                ending_match = self.sentence_endings.match(remaining_text)
                if ending_match:
                    sentence += ending_match.group()
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk + sentence) > self.config.max_chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    chunk_type=ChunkType.SENTENCE,
                    chunk_id=chunk_id,
                    estimated_duration=self._estimate_duration(current_chunk)
                ))
                chunk_id += 1
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                chunk_type=ChunkType.SENTENCE,
                chunk_id=chunk_id,
                estimated_duration=self._estimate_duration(current_chunk)
            ))
        
        # Further split large chunks by phrases
        final_chunks = []
        for chunk in chunks:
            if len(chunk.text) > self.config.max_chunk_size:
                final_chunks.extend(self._split_by_phrases(chunk))
            else:
                final_chunks.append(chunk)
        
        # Add context information
        self._add_context_info(final_chunks, text)
        
        return final_chunks
    
    def _split_by_phrases(self, chunk: TextChunk) -> List[TextChunk]:
        """Split a large chunk by phrase boundaries."""
        phrases = self.phrase_breaks.split(chunk.text)
        sub_chunks = []
        current_text = ""
        chunk_id = chunk.chunk_id
        
        for i, phrase in enumerate(phrases):
            phrase = phrase.strip()
            if not phrase:
                continue
            
            # Add phrase break back
            if i < len(phrases) - 1:
                remaining = chunk.text[chunk.text.find(phrase) + len(phrase):]
                break_match = self.phrase_breaks.match(remaining)
                if break_match:
                    phrase += break_match.group()
            
            if len(current_text + phrase) > self.config.max_chunk_size and current_text:
                sub_chunks.append(TextChunk(
                    text=current_text.strip(),
                    chunk_type=ChunkType.PHRASE,
                    chunk_id=chunk_id,
                    estimated_duration=self._estimate_duration(current_text)
                ))
                chunk_id += 1
                current_text = phrase
            else:
                current_text += " " + phrase if current_text else phrase
        
        if current_text.strip():
            sub_chunks.append(TextChunk(
                text=current_text.strip(),
                chunk_type=ChunkType.PHRASE,
                chunk_id=chunk_id,
                estimated_duration=self._estimate_duration(current_text)
            ))
        
        return sub_chunks
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate audio duration for text."""
        # Rough estimate: ~150 words per minute, ~5 characters per word
        words = len(text) / 5
        duration = (words / 150) * 60  # Convert to seconds
        return max(0.5, duration)  # Minimum 0.5 seconds
    
    def _add_context_info(self, chunks: List[TextChunk], full_text: str) -> None:
        """Add context information to chunks."""
        for i, chunk in enumerate(chunks):
            # Find chunk position in full text
            chunk_start = full_text.find(chunk.text)
            
            # Add context before
            if i > 0:
                chunk.context_before = chunks[i-1].text[-50:]  # Last 50 chars
            elif chunk_start > 0:
                chunk.context_before = full_text[max(0, chunk_start-50):chunk_start]
            
            # Add context after
            if i < len(chunks) - 1:
                chunk.context_after = chunks[i+1].text[:50]  # First 50 chars
            else:
                chunk_end = chunk_start + len(chunk.text)
                chunk.context_after = full_text[chunk_end:chunk_end+50]


class AudioBuffer:
    """Buffer for seamless audio streaming with crossfading."""
    
    def __init__(self, config: StreamingConfig):
        """Initialize the audio buffer.
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        self.buffer: deque = deque(maxlen=config.buffer_size)
        self.sample_rate = 22050
        self.crossfade_samples = int(config.crossfade_duration_ms * self.sample_rate / 1000)
    
    def add_chunk(self, audio_chunk: AudioChunk) -> None:
        """Add an audio chunk to the buffer."""
        # Apply crossfading if there's a previous chunk
        if self.buffer and len(self.buffer) > 0:
            self._apply_crossfade(audio_chunk)
        
        self.buffer.append(audio_chunk)
    
    def get_next_chunk(self) -> Optional[AudioChunk]:
        """Get the next audio chunk from buffer."""
        if self.buffer:
            return self.buffer.popleft()
        return None
    
    def _apply_crossfade(self, new_chunk: AudioChunk) -> None:
        """Apply crossfading between chunks."""
        if not self.buffer:
            return
        
        prev_chunk = self.buffer[-1]
        
        # Apply fade out to previous chunk
        fade_samples = min(self.crossfade_samples, len(prev_chunk.audio_data) // 4)
        if fade_samples > 0:
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            prev_chunk.audio_data[-fade_samples:] *= fade_out
            prev_chunk.fade_out_samples = fade_samples
        
        # Apply fade in to new chunk
        fade_samples = min(self.crossfade_samples, len(new_chunk.audio_data) // 4)
        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            new_chunk.audio_data[:fade_samples] *= fade_in
            new_chunk.fade_in_samples = fade_samples


class RealtimeStreamingSynthesizer:
    """Real-time streaming TTS synthesizer."""
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize the streaming synthesizer.
        
        Args:
            config: Streaming configuration
        """
        self.config = config or StreamingConfig()
        self.model_manager = ModelManager()
        self.audio_processor = get_audio_processor()
        self.text_preprocessor = TextPreprocessor(use_phonemizer=False)
        self.chunker = ContextAwareChunker(self.config)
        self.audio_buffer = AudioBuffer(self.config)
        
        # Performance tracking
        self.chunk_times = []
        self.total_latency = 0.0
    
    async def stream_synthesis(
        self,
        text: str,
        voice: str = "alloy",
        **kwargs
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream TTS synthesis with real-time chunking.
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            **kwargs: Additional synthesis parameters
            
        Yields:
            AudioChunk objects for streaming playback
        """
        start_time = time.time()
        
        try:
            # Chunk the text
            chunks = self.chunker.chunk_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Select optimal model for streaming
            criteria = ModelSelectionCriteria(
                text_length=len(text),
                quality_requirement=self.config.quality_mode,
                performance_requirement="high",
                max_rtf=0.3,  # Strict RTF for streaming
                voice=voice,
                strategy=ModelSelectionStrategy.FAST
            )
            
            # Process chunks concurrently with limited concurrency
            semaphore = asyncio.Semaphore(2)  # Max 2 concurrent synthesis
            tasks = []
            
            for chunk in chunks:
                task = asyncio.create_task(
                    self._synthesize_chunk_with_semaphore(
                        semaphore, chunk, voice, criteria, **kwargs
                    )
                )
                tasks.append(task)
            
            # Yield chunks as they complete
            first_chunk = True
            for task in asyncio.as_completed(tasks):
                try:
                    audio_chunk = await task
                    
                    if first_chunk:
                        first_chunk_latency = time.time() - start_time
                        logger.info(f"First chunk latency: {first_chunk_latency*1000:.1f}ms")
                        first_chunk = False
                    
                    self.audio_buffer.add_chunk(audio_chunk)
                    
                    # Yield buffered chunks
                    while True:
                        buffered_chunk = self.audio_buffer.get_next_chunk()
                        if buffered_chunk is None:
                            break
                        yield buffered_chunk
                        
                except Exception as e:
                    logger.error(f"Chunk synthesis failed: {e}")
                    continue
            
            # Yield any remaining buffered chunks
            while True:
                buffered_chunk = self.audio_buffer.get_next_chunk()
                if buffered_chunk is None:
                    break
                yield buffered_chunk
            
            total_time = time.time() - start_time
            logger.info(f"Total streaming synthesis time: {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            raise
    
    async def _synthesize_chunk_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        chunk: TextChunk,
        voice: str,
        criteria: ModelSelectionCriteria,
        **kwargs
    ) -> AudioChunk:
        """Synthesize a chunk with concurrency control."""
        async with semaphore:
            return await self._synthesize_chunk(chunk, voice, criteria, **kwargs)
    
    async def _synthesize_chunk(
        self,
        chunk: TextChunk,
        voice: str,
        criteria: ModelSelectionCriteria,
        **kwargs
    ) -> AudioChunk:
        """Synthesize a single text chunk."""
        chunk_start = time.time()
        
        try:
            # Load model with fallback
            model = self.model_manager.load_model_with_fallback(criteria)
            
            # Preprocess text
            processed_text = self.text_preprocessor.preprocess(chunk.text)
            
            # Generate audio
            audio_data = model.generate_speech(
                text=processed_text,
                voice=voice,
                **kwargs
            )
            
            # Ensure correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max() * 0.95
            
            chunk_time = time.time() - chunk_start
            self.chunk_times.append(chunk_time)
            
            audio_chunk = AudioChunk(
                audio_data=audio_data,
                sample_rate=model.get_sample_rate(),
                chunk_id=chunk.chunk_id,
                duration=len(audio_data) / model.get_sample_rate(),
                timestamp=time.time()
            )
            
            logger.debug(f"Synthesized chunk {chunk.chunk_id} in {chunk_time:.3f}s")
            return audio_chunk
            
        except Exception as e:
            logger.error(f"Chunk synthesis failed: {e}")
            # Return silence as fallback
            silence_duration = max(0.5, chunk.estimated_duration)
            silence_samples = int(silence_duration * 22050)
            silence_audio = np.zeros(silence_samples, dtype=np.float32)
            
            return AudioChunk(
                audio_data=silence_audio,
                sample_rate=22050,
                chunk_id=chunk.chunk_id,
                duration=silence_duration,
                timestamp=time.time()
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.chunk_times:
            return {}
        
        return {
            "total_chunks": len(self.chunk_times),
            "average_chunk_time": np.mean(self.chunk_times),
            "max_chunk_time": np.max(self.chunk_times),
            "min_chunk_time": np.min(self.chunk_times),
            "total_synthesis_time": np.sum(self.chunk_times)
        }


# Global instance
_streaming_synthesizer = None


def get_streaming_synthesizer(config: Optional[StreamingConfig] = None) -> RealtimeStreamingSynthesizer:
    """Get the global streaming synthesizer instance."""
    global _streaming_synthesizer
    if _streaming_synthesizer is None:
        _streaming_synthesizer = RealtimeStreamingSynthesizer(config)
    return _streaming_synthesizer
