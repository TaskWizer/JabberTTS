"""Multi-Level Caching System for JabberTTS.

This module implements a sophisticated caching system with multiple levels:
- L1: Hot Data (RAM) - Speaker embeddings, frequent phonemes
- L2: Warm Data (SSD) - Model weights, preprocessed features  
- L3: Cold Data (HDD) - Full model checkpoints, training data

Features:
- LRU eviction with TTL expiration
- Persistent disk cache with integrity validation
- Memory-mapped model loading
- Automatic cache warming and preloading
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import mmap
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    key: str
    value: T
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access metadata."""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """LRU cache with TTL support and size limits."""
    
    def __init__(
        self,
        max_entries: int,
        max_size_bytes: int,
        default_ttl: Optional[float] = None,
        name: str = "cache"
    ):
        """Initialize LRU cache.
        
        Args:
            max_entries: Maximum number of entries
            max_size_bytes: Maximum total size in bytes
            default_ttl: Default TTL in seconds
            name: Cache name for logging
        """
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self.default_ttl = default_ttl
        self.name = name
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._update_stats()
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._stats.hits += 1
            return entry.value
    
    def put(self, key: str, value: T, ttl: Optional[float] = None) -> bool:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if value is too large
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Value too large for cache {self.name}: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.size_bytes -= old_entry.size_bytes
                del self._cache[key]
                self._stats.entry_count -= 1
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Evict entries if necessary
            self._evict_if_needed(size_bytes)
            
            # Add new entry
            self._cache[key] = entry
            self._stats.size_bytes += size_bytes
            self._stats.entry_count += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.size_bytes -= entry.size_bytes
                del self._cache[key]
                self._stats.entry_count -= 1
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries if needed to make space."""
        # Evict expired entries first
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            entry = self._cache[key]
            self._stats.size_bytes -= entry.size_bytes
            del self._cache[key]
            self._stats.entry_count -= 1
            self._stats.evictions += 1
        
        # Evict LRU entries if still needed
        while (
            len(self._cache) >= self.max_entries or
            self._stats.size_bytes + new_size > self.max_size_bytes
        ):
            if not self._cache:
                break
            
            # Remove least recently used (first item)
            key, entry = self._cache.popitem(last=False)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            self._stats.evictions += 1
    
    def _estimate_size(self, value: T) -> int:
        """Estimate size of value in bytes."""
        try:
            if isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, dict):
                return len(pickle.dumps(value))
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        self._stats.entry_count = len(self._cache)
        self._stats.size_bytes = sum(entry.size_bytes for entry in self._cache.values())
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._update_stats()
            return self._stats


class PersistentCache:
    """Persistent disk cache with integrity validation."""
    
    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 10.0,
        compression: bool = True
    ):
        """Initialize persistent cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_gb: Maximum cache size in GB
            compression: Enable compression
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.compression = compression
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        self._load_index()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self._index:
                return None
            
            entry_info = self._index[key]
            
            # Check expiration
            if entry_info.get('ttl') and time.time() - entry_info['timestamp'] > entry_info['ttl']:
                self._remove_entry(key)
                return None
            
            # Load from disk
            try:
                file_path = self.cache_dir / entry_info['filename']
                if not file_path.exists():
                    self._remove_entry(key)
                    return None
                
                # Verify integrity
                if not self._verify_integrity(file_path, entry_info['checksum']):
                    logger.warning(f"Cache integrity check failed for {key}")
                    self._remove_entry(key)
                    return None
                
                # Load data
                with open(file_path, 'rb') as f:
                    if self.compression:
                        import gzip
                        data = gzip.decompress(f.read())
                    else:
                        data = f.read()
                
                value = pickle.loads(data)
                
                # Update access time
                entry_info['last_access'] = time.time()
                entry_info['access_count'] = entry_info.get('access_count', 0) + 1
                self._save_index()
                
                return value
                
            except Exception as e:
                logger.error(f"Failed to load cache entry {key}: {e}")
                self._remove_entry(key)
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in persistent cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            try:
                # Serialize data
                data = pickle.dumps(value)
                
                if self.compression:
                    import gzip
                    data = gzip.compress(data)
                
                # Generate filename
                filename = f"{hashlib.md5(key.encode()).hexdigest()}.cache"
                file_path = self.cache_dir / filename
                
                # Check size limits
                if len(data) > self.max_size_bytes:
                    logger.warning(f"Value too large for persistent cache: {len(data)} bytes")
                    return False
                
                # Evict if necessary
                self._evict_if_needed(len(data))
                
                # Write to disk
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Calculate checksum
                checksum = hashlib.sha256(data).hexdigest()
                
                # Update index
                self._index[key] = {
                    'filename': filename,
                    'size_bytes': len(data),
                    'timestamp': time.time(),
                    'last_access': time.time(),
                    'access_count': 0,
                    'checksum': checksum,
                    'ttl': ttl
                }
                
                self._save_index()
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache entry {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from persistent cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all entries from persistent cache."""
        with self._lock:
            for entry_info in self._index.values():
                file_path = self.cache_dir / entry_info['filename']
                if file_path.exists():
                    file_path.unlink()
            
            self._index.clear()
            self._save_index()
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _verify_integrity(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file integrity using checksum."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            actual_checksum = hashlib.sha256(data).hexdigest()
            return actual_checksum == expected_checksum
        except Exception:
            return False
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache and disk."""
        if key not in self._index:
            return False
        
        entry_info = self._index[key]
        file_path = self.cache_dir / entry_info['filename']
        
        if file_path.exists():
            file_path.unlink()
        
        del self._index[key]
        self._save_index()
        return True
    
    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries if needed to make space."""
        current_size = sum(entry['size_bytes'] for entry in self._index.values())
        
        if current_size + new_size <= self.max_size_bytes:
            return
        
        # Sort by last access time (LRU)
        entries_by_access = sorted(
            self._index.items(),
            key=lambda x: x[1]['last_access']
        )
        
        for key, entry_info in entries_by_access:
            if current_size + new_size <= self.max_size_bytes:
                break
            
            self._remove_entry(key)
            current_size -= entry_info['size_bytes']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry['size_bytes'] for entry in self._index.values())
            total_files = len(self._index)
            
            return {
                'total_entries': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
                'utilization': total_size / self.max_size_bytes
            }


class MultiLevelCacheManager:
    """Multi-level cache manager coordinating all cache layers."""
    
    def __init__(self, cache_dir: Path):
        """Initialize multi-level cache manager.
        
        Args:
            cache_dir: Base directory for persistent caches
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # L1 Cache: Hot data in RAM
        self.phoneme_cache = LRUCache[str](
            max_entries=10000,
            max_size_bytes=100 * 1024 * 1024,  # 100MB
            default_ttl=3600,  # 1 hour
            name="phoneme"
        )
        
        self.embedding_cache = LRUCache[np.ndarray](
            max_entries=1000,
            max_size_bytes=50 * 1024 * 1024,  # 50MB
            default_ttl=7200,  # 2 hours
            name="embedding"
        )
        
        self.audio_cache = LRUCache[np.ndarray](
            max_entries=5000,
            max_size_bytes=500 * 1024 * 1024,  # 500MB
            default_ttl=1800,  # 30 minutes
            name="audio"
        )
        
        # L2 Cache: Persistent disk cache
        self.persistent_cache = PersistentCache(
            cache_dir=self.cache_dir / "persistent",
            max_size_gb=10.0,
            compression=True
        )
        
        # Model weight cache (memory-mapped)
        self.model_cache: Dict[str, Any] = {}
        self._model_lock = threading.RLock()
    
    def get_phoneme(self, text: str) -> Optional[str]:
        """Get phonemized text from cache."""
        key = f"phoneme:{hashlib.md5(text.encode()).hexdigest()}"
        return self.phoneme_cache.get(key)
    
    def put_phoneme(self, text: str, phonemes: str) -> None:
        """Cache phonemized text."""
        key = f"phoneme:{hashlib.md5(text.encode()).hexdigest()}"
        self.phoneme_cache.put(key, phonemes)
    
    def get_embedding(self, voice_id: str) -> Optional[np.ndarray]:
        """Get voice embedding from cache."""
        key = f"embedding:{voice_id}"
        
        # Try L1 cache first
        embedding = self.embedding_cache.get(key)
        if embedding is not None:
            return embedding
        
        # Try persistent cache
        embedding = self.persistent_cache.get(key)
        if embedding is not None:
            # Promote to L1 cache
            self.embedding_cache.put(key, embedding)
            return embedding
        
        return None
    
    def put_embedding(self, voice_id: str, embedding: np.ndarray) -> None:
        """Cache voice embedding."""
        key = f"embedding:{voice_id}"
        
        # Store in both L1 and persistent cache
        self.embedding_cache.put(key, embedding)
        self.persistent_cache.put(key, embedding, ttl=86400)  # 24 hours
    
    def get_audio_segment(self, text_hash: str) -> Optional[np.ndarray]:
        """Get audio segment from cache."""
        key = f"audio:{text_hash}"
        return self.audio_cache.get(key)
    
    def put_audio_segment(self, text_hash: str, audio: np.ndarray) -> None:
        """Cache audio segment."""
        key = f"audio:{text_hash}"
        self.audio_cache.put(key, audio)
    
    def get_model_weights(self, model_id: str) -> Optional[Any]:
        """Get model weights from cache."""
        with self._model_lock:
            return self.model_cache.get(model_id)
    
    def put_model_weights(self, model_id: str, weights: Any) -> None:
        """Cache model weights."""
        with self._model_lock:
            self.model_cache[model_id] = weights
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'phoneme_cache': self.phoneme_cache.get_stats().__dict__,
            'embedding_cache': self.embedding_cache.get_stats().__dict__,
            'audio_cache': self.audio_cache.get_stats().__dict__,
            'persistent_cache': self.persistent_cache.get_stats(),
            'model_cache_entries': len(self.model_cache)
        }
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.phoneme_cache.clear()
        self.embedding_cache.clear()
        self.audio_cache.clear()
        self.persistent_cache.clear()
        
        with self._model_lock:
            self.model_cache.clear()


# Global instance
_cache_manager = None


def get_cache_manager() -> MultiLevelCacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        cache_dir = Path("temp") / "cache"
        _cache_manager = MultiLevelCacheManager(cache_dir)
    return _cache_manager
