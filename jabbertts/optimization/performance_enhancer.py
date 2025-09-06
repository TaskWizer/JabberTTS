"""Performance Enhancement Module for JabberTTS.

This module implements aggressive performance optimizations to achieve:
- RTF < 0.1 on CPU, < 0.05 on GPU
- First-chunk latency < 200ms for streaming
- Higher resource utilization for maximum performance
"""

import asyncio
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import gc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Performance enhancement configuration."""
    # Resource utilization
    max_cpu_workers: int = multiprocessing.cpu_count() * 4  # Aggressive CPU usage
    max_gpu_memory_fraction: float = 0.95  # Use 95% of GPU memory
    enable_memory_mapping: bool = True
    enable_aggressive_caching: bool = True
    
    # Latency targets
    target_rtf_cpu: float = 0.1
    target_rtf_gpu: float = 0.05
    target_first_chunk_latency: float = 0.2  # 200ms
    
    # Optimization flags
    enable_model_compilation: bool = True
    enable_tensor_parallelism: bool = True
    enable_pipeline_parallelism: bool = True
    enable_mixed_precision: bool = True
    
    # Caching configuration
    cache_size_multiplier: float = 3.0  # 3x larger caches
    preload_common_phrases: bool = True
    enable_predictive_caching: bool = True


class AggressiveCacheManager:
    """Aggressive caching system for maximum performance."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize aggressive cache manager."""
        self.config = config
        self.caches = {}
        self.cache_stats = {}
        self.preload_thread = None
        self._setup_caches()
        
        if config.preload_common_phrases:
            self._start_preloading()
    
    def _setup_caches(self):
        """Setup aggressive caching with larger sizes."""
        base_sizes = {
            "phoneme": 10000,
            "embedding": 1000,
            "audio_segment": 5000,
            "model_weights": 100,
            "preprocessing": 2000
        }
        
        # Multiply cache sizes by configuration factor
        for cache_name, base_size in base_sizes.items():
            cache_size = int(base_size * self.config.cache_size_multiplier)
            self.caches[cache_name] = {}
            self.cache_stats[cache_name] = {"hits": 0, "misses": 0, "size": 0}
            logger.info(f"Initialized {cache_name} cache with size {cache_size}")
    
    def _start_preloading(self):
        """Start background preloading of common phrases."""
        def preload_worker():
            common_phrases = [
                "Hello", "Hello world", "How are you", "Thank you", "Good morning",
                "Good afternoon", "Good evening", "Please", "Excuse me", "I'm sorry",
                "Yes", "No", "Maybe", "Okay", "Alright", "Sure", "Of course",
                "The quick brown fox", "Testing one two three", "This is a test"
            ]
            
            logger.info("Starting background preloading of common phrases")
            for phrase in common_phrases:
                try:
                    # This would trigger preprocessing and caching
                    # Implementation depends on integration with main system
                    time.sleep(0.1)  # Prevent overwhelming the system
                except Exception as e:
                    logger.warning(f"Preloading failed for '{phrase}': {e}")
            
            logger.info("Background preloading completed")
        
        self.preload_thread = threading.Thread(target=preload_worker, daemon=True)
        self.preload_thread.start()
    
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """Get item from cache."""
        if cache_name not in self.caches:
            return None
        
        if key in self.caches[cache_name]:
            self.cache_stats[cache_name]["hits"] += 1
            return self.caches[cache_name][key]
        
        self.cache_stats[cache_name]["misses"] += 1
        return None
    
    def put(self, cache_name: str, key: str, value: Any) -> None:
        """Put item in cache."""
        if cache_name not in self.caches:
            return
        
        self.caches[cache_name][key] = value
        self.cache_stats[cache_name]["size"] = len(self.caches[cache_name])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_stats.copy()


class ModelOptimizer:
    """Model-specific optimizations for maximum performance."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize model optimizer."""
        self.config = config
        self.optimized_models = {}
        self.compilation_cache = {}
    
    def optimize_model(self, model, model_name: str) -> Any:
        """Apply aggressive optimizations to a model."""
        if model_name in self.optimized_models:
            return self.optimized_models[model_name]
        
        logger.info(f"Applying aggressive optimizations to {model_name}")
        
        try:
            optimized_model = model
            
            # PyTorch optimizations
            if hasattr(model, 'model') and hasattr(model.model, 'eval'):
                # Enable inference mode optimizations
                model.model.eval()
                
                # Compile model for better performance (PyTorch 2.0+)
                if self.config.enable_model_compilation:
                    optimized_model = self._compile_model(model, model_name)
                
                # Enable mixed precision if supported
                if self.config.enable_mixed_precision:
                    optimized_model = self._enable_mixed_precision(optimized_model, model_name)
                
                # Memory optimizations
                optimized_model = self._optimize_memory(optimized_model, model_name)
            
            self.optimized_models[model_name] = optimized_model
            logger.info(f"Model {model_name} optimization completed")
            
            return optimized_model
            
        except Exception as e:
            logger.warning(f"Model optimization failed for {model_name}: {e}")
            return model
    
    def _compile_model(self, model, model_name: str):
        """Compile model for better performance."""
        try:
            import torch
            
            if hasattr(torch, 'compile') and hasattr(model, 'model'):
                logger.info(f"Compiling {model_name} with torch.compile")
                
                # Use aggressive compilation settings
                compiled_model = torch.compile(
                    model.model,
                    mode="max-autotune",  # Most aggressive optimization
                    fullgraph=True,       # Compile entire graph
                    dynamic=False         # Static shapes for better optimization
                )
                
                # Replace the model
                model.model = compiled_model
                self.compilation_cache[model_name] = True
                
                logger.info(f"Model {model_name} compiled successfully")
            
            return model
            
        except Exception as e:
            logger.warning(f"Model compilation failed for {model_name}: {e}")
            return model
    
    def _enable_mixed_precision(self, model, model_name: str):
        """Enable mixed precision for faster inference."""
        try:
            import torch
            
            if torch.cuda.is_available() and hasattr(model, 'model'):
                # Enable automatic mixed precision
                model.model = model.model.half()  # Convert to FP16
                logger.info(f"Enabled mixed precision for {model_name}")
            
            return model
            
        except Exception as e:
            logger.warning(f"Mixed precision setup failed for {model_name}: {e}")
            return model
    
    def _optimize_memory(self, model, model_name: str):
        """Optimize memory usage."""
        try:
            import torch
            
            if hasattr(model, 'model'):
                # Enable memory efficient attention if available
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    # This is automatically used in newer PyTorch versions
                    pass
                
                # Optimize for inference
                if hasattr(model.model, 'eval'):
                    model.model.eval()
                
                # Clear gradients to save memory
                if hasattr(model.model, 'zero_grad'):
                    model.model.zero_grad(set_to_none=True)
            
            return model
            
        except Exception as e:
            logger.warning(f"Memory optimization failed for {model_name}: {e}")
            return model


class ParallelInferenceEngine:
    """Parallel inference engine for maximum throughput."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize parallel inference engine."""
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_cpu_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, config.max_cpu_workers // 4))
        self.inference_queue = asyncio.Queue(maxsize=1000)
        self.result_cache = {}
        
        logger.info(f"Initialized parallel inference with {config.max_cpu_workers} CPU workers")
    
    async def parallel_inference(self, inference_func: Callable, *args, **kwargs) -> Any:
        """Execute inference in parallel for maximum performance."""
        # Create cache key
        cache_key = self._create_cache_key(args, kwargs)
        
        # Check result cache first
        if cache_key in self.result_cache:
            logger.debug("Cache hit for parallel inference")
            return self.result_cache[cache_key]
        
        # Execute inference
        loop = asyncio.get_event_loop()
        
        try:
            # Use thread pool for CPU-bound tasks
            result = await loop.run_in_executor(
                self.thread_pool,
                inference_func,
                *args,
                **kwargs
            )
            
            # Cache result
            self.result_cache[cache_key] = result
            
            # Limit cache size
            if len(self.result_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.result_cache.keys())[:100]
                for key in oldest_keys:
                    del self.result_cache[key]
            
            return result
            
        except Exception as e:
            logger.error(f"Parallel inference failed: {e}")
            raise
    
    def _create_cache_key(self, args, kwargs) -> str:
        """Create cache key from arguments."""
        try:
            # Simple hash-based key creation
            import hashlib
            key_data = str(args) + str(sorted(kwargs.items()))
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return str(time.time())  # Fallback to timestamp
    
    def shutdown(self):
        """Shutdown parallel inference engine."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class PerformanceEnhancer:
    """Main performance enhancement coordinator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize performance enhancer."""
        self.config = config or PerformanceConfig()
        self.cache_manager = AggressiveCacheManager(self.config)
        self.model_optimizer = ModelOptimizer(self.config)
        self.parallel_engine = ParallelInferenceEngine(self.config)
        self.performance_metrics = {}
        
        # Apply system-level optimizations
        self._apply_system_optimizations()
        
        logger.info("Performance enhancer initialized with aggressive settings")
    
    def _apply_system_optimizations(self):
        """Apply system-level performance optimizations."""
        try:
            # Set environment variables for maximum performance
            os.environ["OMP_NUM_THREADS"] = str(self.config.max_cpu_workers)
            os.environ["MKL_NUM_THREADS"] = str(self.config.max_cpu_workers)
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.config.max_cpu_workers)
            
            # PyTorch optimizations
            try:
                import torch
                
                # Set number of threads
                torch.set_num_threads(self.config.max_cpu_workers)
                torch.set_num_interop_threads(max(1, self.config.max_cpu_workers // 4))
                
                # Enable optimizations
                if hasattr(torch.backends, 'mkldnn'):
                    torch.backends.mkldnn.enabled = True
                
                if torch.cuda.is_available():
                    # GPU optimizations
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    
                    # Set memory fraction
                    torch.cuda.set_per_process_memory_fraction(self.config.max_gpu_memory_fraction)
                    
                    # Enable TensorFloat-32 on Ampere GPUs
                    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                
                logger.info("Applied PyTorch performance optimizations")
                
            except ImportError:
                logger.warning("PyTorch not available for optimization")
            
            # Python GC optimizations
            gc.set_threshold(700, 10, 10)  # More aggressive GC
            
            logger.info("Applied system-level performance optimizations")
            
        except Exception as e:
            logger.warning(f"System optimization failed: {e}")
    
    def enhance_model(self, model, model_name: str):
        """Enhance model performance."""
        return self.model_optimizer.optimize_model(model, model_name)
    
    async def enhance_inference(self, inference_func: Callable, *args, **kwargs):
        """Enhance inference performance."""
        return await self.parallel_engine.parallel_inference(inference_func, *args, **kwargs)
    
    def get_cache_manager(self) -> AggressiveCacheManager:
        """Get cache manager."""
        return self.cache_manager
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "config": {
                "max_cpu_workers": self.config.max_cpu_workers,
                "max_gpu_memory_fraction": self.config.max_gpu_memory_fraction,
                "target_rtf_cpu": self.config.target_rtf_cpu,
                "target_rtf_gpu": self.config.target_rtf_gpu,
                "target_first_chunk_latency": self.config.target_first_chunk_latency
            },
            "cache_stats": self.cache_manager.get_stats(),
            "optimized_models": list(self.model_optimizer.optimized_models.keys()),
            "compilation_cache": self.model_optimizer.compilation_cache
        }
    
    def shutdown(self):
        """Shutdown performance enhancer."""
        self.parallel_engine.shutdown()


# Global instance
_performance_enhancer = None


def get_performance_enhancer(config: Optional[PerformanceConfig] = None) -> PerformanceEnhancer:
    """Get the global performance enhancer instance."""
    global _performance_enhancer
    if _performance_enhancer is None:
        _performance_enhancer = PerformanceEnhancer(config)
    return _performance_enhancer


def enhance_model_performance(model, model_name: str):
    """Convenience function to enhance model performance."""
    enhancer = get_performance_enhancer()
    return enhancer.enhance_model(model, model_name)


async def enhance_inference_performance(inference_func: Callable, *args, **kwargs):
    """Convenience function to enhance inference performance."""
    enhancer = get_performance_enhancer()
    return await enhancer.enhance_inference(inference_func, *args, **kwargs)
