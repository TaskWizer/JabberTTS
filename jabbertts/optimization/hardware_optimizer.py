"""Hardware-Specific Optimizations for JabberTTS.

This module implements hardware-specific optimizations including:
- CPU SIMD optimizations (AVX/AVX2/NEON)
- GPU mixed precision (FP16/INT8) and memory optimization
- ONNX Runtime provider tuning and session configuration
- Memory management with lazy loading and Python GC optimization
"""

import gc
import logging
import os
import platform
import psutil
import threading
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Hardware type classification."""
    CPU_INTEL = "cpu_intel"
    CPU_AMD = "cpu_amd"
    CPU_ARM64 = "cpu_arm64"
    GPU_NVIDIA = "gpu_nvidia"
    GPU_AMD = "gpu_amd"
    GPU_INTEL = "gpu_intel"
    UNKNOWN = "unknown"


@dataclass
class HardwareInfo:
    """Hardware information and capabilities."""
    hardware_type: HardwareType
    cpu_count: int
    memory_gb: float
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = False
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: Optional[str] = None
    supports_mixed_precision: bool = False
    numa_nodes: int = 1


@dataclass
class OptimizationConfig:
    """Hardware optimization configuration."""
    enable_simd: bool = True
    enable_mixed_precision: bool = True
    enable_memory_mapping: bool = True
    enable_numa_optimization: bool = True
    gc_threshold_multiplier: float = 2.0
    memory_pool_size_mb: int = 1024
    onnx_optimization_level: str = "all"
    thread_affinity: bool = False


class CPUOptimizer:
    """CPU-specific optimizations."""
    
    def __init__(self, hardware_info: HardwareInfo):
        """Initialize CPU optimizer.
        
        Args:
            hardware_info: Hardware information
        """
        self.hardware_info = hardware_info
        self._original_gc_thresholds = None
    
    def optimize_cpu_performance(self, config: OptimizationConfig) -> None:
        """Apply CPU-specific optimizations.
        
        Args:
            config: Optimization configuration
        """
        logger.info("Applying CPU optimizations...")
        
        # SIMD optimizations
        if config.enable_simd:
            self._enable_simd_optimizations()
        
        # NUMA optimizations
        if config.enable_numa_optimization and self.hardware_info.numa_nodes > 1:
            self._optimize_numa_topology()
        
        # Thread affinity
        if config.thread_affinity:
            self._set_thread_affinity()
        
        # Python GC optimization
        self._optimize_garbage_collection(config.gc_threshold_multiplier)
        
        # Memory optimization
        self._optimize_memory_allocation()
    
    def _enable_simd_optimizations(self) -> None:
        """Enable SIMD optimizations based on CPU capabilities."""
        try:
            # Set environment variables for optimized libraries
            if self.hardware_info.has_avx512:
                os.environ["OMP_NUM_THREADS"] = str(self.hardware_info.cpu_count)
                os.environ["MKL_NUM_THREADS"] = str(self.hardware_info.cpu_count)
                os.environ["OPENBLAS_NUM_THREADS"] = str(self.hardware_info.cpu_count)
                logger.info("Enabled AVX-512 optimizations")
            
            elif self.hardware_info.has_avx2:
                os.environ["OMP_NUM_THREADS"] = str(self.hardware_info.cpu_count)
                os.environ["MKL_NUM_THREADS"] = str(self.hardware_info.cpu_count)
                logger.info("Enabled AVX2 optimizations")
            
            elif self.hardware_info.has_neon:
                # ARM NEON optimizations
                os.environ["OMP_NUM_THREADS"] = str(min(4, self.hardware_info.cpu_count))
                logger.info("Enabled NEON optimizations")
            
            # PyTorch CPU optimizations
            try:
                import torch
                torch.set_num_threads(self.hardware_info.cpu_count)
                torch.set_num_interop_threads(max(1, self.hardware_info.cpu_count // 4))
                
                if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
                    torch.backends.mkldnn.enabled = True
                    logger.info("Enabled MKL-DNN optimizations")
                
            except ImportError:
                pass
            
        except Exception as e:
            logger.warning(f"Failed to enable SIMD optimizations: {e}")
    
    def _optimize_numa_topology(self) -> None:
        """Optimize for NUMA topology."""
        try:
            # Set memory policy for NUMA
            os.environ["NUMA_POLICY"] = "interleave"
            
            # Bind to local NUMA node if possible
            try:
                import numa
                if numa.available():
                    numa.set_preferred(0)  # Prefer node 0
                    logger.info("Optimized NUMA topology")
            except ImportError:
                logger.debug("NUMA library not available")
                
        except Exception as e:
            logger.warning(f"Failed to optimize NUMA topology: {e}")
    
    def _set_thread_affinity(self) -> None:
        """Set thread affinity for better cache locality."""
        try:
            import psutil
            process = psutil.Process()
            
            # Set CPU affinity to all available cores
            available_cpus = list(range(self.hardware_info.cpu_count))
            process.cpu_affinity(available_cpus)
            
            logger.info(f"Set thread affinity to CPUs: {available_cpus}")
            
        except Exception as e:
            logger.warning(f"Failed to set thread affinity: {e}")
    
    def _optimize_garbage_collection(self, multiplier: float) -> None:
        """Optimize Python garbage collection."""
        try:
            # Store original thresholds
            self._original_gc_thresholds = gc.get_threshold()
            
            # Increase GC thresholds to reduce frequency
            new_thresholds = tuple(int(t * multiplier) for t in self._original_gc_thresholds)
            gc.set_threshold(*new_thresholds)
            
            # Disable GC during critical sections (can be re-enabled manually)
            gc.disable()
            
            logger.info(f"Optimized GC thresholds: {self._original_gc_thresholds} -> {new_thresholds}")
            
        except Exception as e:
            logger.warning(f"Failed to optimize garbage collection: {e}")
    
    def _optimize_memory_allocation(self) -> None:
        """Optimize memory allocation patterns."""
        try:
            # Set memory allocation strategy
            if platform.system() == "Linux":
                # Use transparent huge pages if available
                try:
                    with open("/sys/kernel/mm/transparent_hugepage/enabled", "r") as f:
                        if "always" in f.read():
                            logger.info("Transparent huge pages enabled")
                except:
                    pass
            
            # Set malloc behavior
            os.environ["MALLOC_TRIM_THRESHOLD_"] = "131072"  # 128KB
            os.environ["MALLOC_MMAP_THRESHOLD_"] = "131072"   # 128KB
            
        except Exception as e:
            logger.warning(f"Failed to optimize memory allocation: {e}")
    
    def restore_defaults(self) -> None:
        """Restore default settings."""
        if self._original_gc_thresholds:
            gc.set_threshold(*self._original_gc_thresholds)
            gc.enable()


class GPUOptimizer:
    """GPU-specific optimizations."""
    
    def __init__(self, hardware_info: HardwareInfo):
        """Initialize GPU optimizer.
        
        Args:
            hardware_info: Hardware information
        """
        self.hardware_info = hardware_info
    
    def optimize_gpu_performance(self, config: OptimizationConfig) -> None:
        """Apply GPU-specific optimizations.
        
        Args:
            config: Optimization configuration
        """
        logger.info("Applying GPU optimizations...")
        
        # PyTorch GPU optimizations
        self._optimize_pytorch_gpu()
        
        # Mixed precision
        if config.enable_mixed_precision and self.hardware_info.supports_mixed_precision:
            self._enable_mixed_precision()
        
        # Memory optimization
        self._optimize_gpu_memory()
        
        # CUDA-specific optimizations
        if self.hardware_info.hardware_type == HardwareType.GPU_NVIDIA:
            self._optimize_cuda()
    
    def _optimize_pytorch_gpu(self) -> None:
        """Optimize PyTorch GPU settings."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Enable cuDNN benchmark mode
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Enable TensorFloat-32 (TF32) on Ampere GPUs
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                # Set memory fraction
                memory_fraction = min(0.9, self.hardware_info.gpu_memory_gb / 8.0)
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                
                logger.info("Optimized PyTorch GPU settings")
            
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to optimize PyTorch GPU: {e}")
    
    def _enable_mixed_precision(self) -> None:
        """Enable mixed precision training/inference."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Enable automatic mixed precision
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                
                logger.info("Enabled mixed precision optimizations")
            
        except Exception as e:
            logger.warning(f"Failed to enable mixed precision: {e}")
    
    def _optimize_gpu_memory(self) -> None:
        """Optimize GPU memory usage."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Set memory pool configuration
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                
                # Clear cache
                torch.cuda.empty_cache()
                
                # Set memory growth (if using TensorFlow)
                try:
                    import tensorflow as tf
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info("Enabled TensorFlow GPU memory growth")
                except ImportError:
                    pass
                
                logger.info("Optimized GPU memory settings")
            
        except Exception as e:
            logger.warning(f"Failed to optimize GPU memory: {e}")
    
    def _optimize_cuda(self) -> None:
        """CUDA-specific optimizations."""
        try:
            # Set CUDA environment variables
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async kernel launches
            os.environ["CUDA_CACHE_DISABLE"] = "0"    # Enable kernel cache
            
            # Optimize CUDA streams
            import torch
            if torch.cuda.is_available():
                # Create high priority stream
                stream = torch.cuda.Stream(priority=-1)
                torch.cuda.set_stream(stream)
                
                logger.info("Applied CUDA optimizations")
            
        except Exception as e:
            logger.warning(f"Failed to apply CUDA optimizations: {e}")


class ONNXOptimizer:
    """ONNX Runtime optimizations."""
    
    def __init__(self, hardware_info: HardwareInfo):
        """Initialize ONNX optimizer.
        
        Args:
            hardware_info: Hardware information
        """
        self.hardware_info = hardware_info
    
    def get_optimized_providers(self, config: OptimizationConfig) -> List[Tuple[str, Dict[str, Any]]]:
        """Get optimized ONNX Runtime providers.
        
        Args:
            config: Optimization configuration
            
        Returns:
            List of (provider_name, provider_options) tuples
        """
        providers = []
        
        # GPU providers (highest priority)
        if self.hardware_info.hardware_type == HardwareType.GPU_NVIDIA:
            providers.append(self._get_tensorrt_provider(config))
            providers.append(self._get_cuda_provider(config))
        
        elif self.hardware_info.hardware_type == HardwareType.GPU_AMD:
            providers.append(self._get_rocm_provider(config))
        
        elif self.hardware_info.hardware_type == HardwareType.GPU_INTEL:
            providers.append(self._get_openvino_provider(config))
        
        # CPU providers (fallback)
        providers.append(self._get_cpu_provider(config))
        
        return providers
    
    def _get_tensorrt_provider(self, config: OptimizationConfig) -> Tuple[str, Dict[str, Any]]:
        """Get TensorRT execution provider configuration."""
        options = {
            'trt_max_workspace_size': int(2 * 1024 * 1024 * 1024),  # 2GB
            'trt_fp16_enable': config.enable_mixed_precision,
            'trt_int8_enable': False,  # Requires calibration
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './temp/tensorrt_cache',
            'trt_timing_cache_enable': True,
            'trt_force_sequential_engine_build': False
        }
        
        return ('TensorrtExecutionProvider', options)
    
    def _get_cuda_provider(self, config: OptimizationConfig) -> Tuple[str, Dict[str, Any]]:
        """Get CUDA execution provider configuration."""
        options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': int(self.hardware_info.gpu_memory_gb * 0.8 * 1024 * 1024 * 1024),
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
            'cudnn_conv_use_max_workspace': True
        }
        
        return ('CUDAExecutionProvider', options)
    
    def _get_rocm_provider(self, config: OptimizationConfig) -> Tuple[str, Dict[str, Any]]:
        """Get ROCm execution provider configuration."""
        options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': int(self.hardware_info.gpu_memory_gb * 0.8 * 1024 * 1024 * 1024),
            'miopen_conv_exhaustive_search': True
        }
        
        return ('ROCMExecutionProvider', options)
    
    def _get_openvino_provider(self, config: OptimizationConfig) -> Tuple[str, Dict[str, Any]]:
        """Get OpenVINO execution provider configuration."""
        options = {
            'device_type': 'GPU_FP16' if config.enable_mixed_precision else 'GPU',
            'precision': 'FP16' if config.enable_mixed_precision else 'FP32',
            'num_of_threads': self.hardware_info.cpu_count,
            'cache_dir': './temp/openvino_cache'
        }
        
        return ('OpenVINOExecutionProvider', options)
    
    def _get_cpu_provider(self, config: OptimizationConfig) -> Tuple[str, Dict[str, Any]]:
        """Get CPU execution provider configuration."""
        options = {
            'intra_op_num_threads': self.hardware_info.cpu_count,
            'inter_op_num_threads': max(1, self.hardware_info.cpu_count // 4),
            'enable_cpu_mem_arena': True,
            'enable_memory_pattern': True,
            'execution_mode': 'parallel'
        }
        
        return ('CPUExecutionProvider', options)
    
    def get_session_options(self, config: OptimizationConfig):
        """Get optimized ONNX Runtime session options.
        
        Args:
            config: Optimization configuration
            
        Returns:
            ONNX Runtime SessionOptions
        """
        try:
            import onnxruntime as ort
            
            session_options = ort.SessionOptions()
            
            # Graph optimization
            if config.onnx_optimization_level == "all":
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            elif config.onnx_optimization_level == "extended":
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            else:
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            
            # Threading
            session_options.intra_op_num_threads = self.hardware_info.cpu_count
            session_options.inter_op_num_threads = max(1, self.hardware_info.cpu_count // 4)
            
            # Memory optimization
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
            
            # Execution mode
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            # Profiling (disable in production)
            session_options.enable_profiling = False
            
            return session_options
            
        except ImportError:
            logger.warning("ONNX Runtime not available")
            return None


class HardwareDetector:
    """Hardware detection and capability analysis."""
    
    @staticmethod
    def detect_hardware() -> HardwareInfo:
        """Detect hardware capabilities.
        
        Returns:
            Hardware information
        """
        # CPU detection
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Architecture detection
        machine = platform.machine().lower()
        if machine in ['x86_64', 'amd64']:
            if 'intel' in platform.processor().lower():
                hardware_type = HardwareType.CPU_INTEL
            else:
                hardware_type = HardwareType.CPU_AMD
        elif machine in ['arm64', 'aarch64']:
            hardware_type = HardwareType.CPU_ARM64
        else:
            hardware_type = HardwareType.UNKNOWN
        
        # CPU features
        has_avx2 = HardwareDetector._check_cpu_feature('avx2')
        has_avx512 = HardwareDetector._check_cpu_feature('avx512')
        has_neon = machine in ['arm64', 'aarch64']
        
        # GPU detection
        gpu_memory_gb = 0.0
        gpu_compute_capability = None
        supports_mixed_precision = False
        
        try:
            import torch
            if torch.cuda.is_available():
                hardware_type = HardwareType.GPU_NVIDIA
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_compute_capability = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
                supports_mixed_precision = torch.cuda.get_device_capability(0)[0] >= 7  # Volta+
        except ImportError:
            pass
        
        # NUMA detection
        numa_nodes = 1
        try:
            numa_nodes = len(psutil.cpu_count(logical=False) or [1])
        except:
            pass
        
        return HardwareInfo(
            hardware_type=hardware_type,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            has_avx2=has_avx2,
            has_avx512=has_avx512,
            has_neon=has_neon,
            gpu_memory_gb=gpu_memory_gb,
            gpu_compute_capability=gpu_compute_capability,
            supports_mixed_precision=supports_mixed_precision,
            numa_nodes=numa_nodes
        )
    
    @staticmethod
    def _check_cpu_feature(feature: str) -> bool:
        """Check if CPU supports a specific feature."""
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    return feature in cpuinfo
            # Add Windows/macOS detection if needed
            return False
        except:
            return False


class HardwareOptimizationManager:
    """Main hardware optimization manager."""
    
    def __init__(self):
        """Initialize hardware optimization manager."""
        self.hardware_info = HardwareDetector.detect_hardware()
        self.cpu_optimizer = CPUOptimizer(self.hardware_info)
        self.gpu_optimizer = GPUOptimizer(self.hardware_info)
        self.onnx_optimizer = ONNXOptimizer(self.hardware_info)
        self._optimizations_applied = False
    
    def apply_optimizations(self, config: Optional[OptimizationConfig] = None) -> None:
        """Apply all hardware optimizations.
        
        Args:
            config: Optimization configuration
        """
        if self._optimizations_applied:
            return
        
        config = config or OptimizationConfig()
        
        logger.info(f"Applying optimizations for {self.hardware_info.hardware_type.value}")
        
        # CPU optimizations
        self.cpu_optimizer.optimize_cpu_performance(config)
        
        # GPU optimizations
        if self.hardware_info.gpu_memory_gb > 0:
            self.gpu_optimizer.optimize_gpu_performance(config)
        
        self._optimizations_applied = True
        logger.info("Hardware optimizations applied successfully")
    
    def get_onnx_config(self, config: Optional[OptimizationConfig] = None) -> Dict[str, Any]:
        """Get ONNX Runtime configuration.
        
        Args:
            config: Optimization configuration
            
        Returns:
            ONNX configuration dictionary
        """
        config = config or OptimizationConfig()
        
        providers = self.onnx_optimizer.get_optimized_providers(config)
        session_options = self.onnx_optimizer.get_session_options(config)
        
        return {
            'providers': providers,
            'session_options': session_options
        }
    
    def get_hardware_info(self) -> HardwareInfo:
        """Get hardware information."""
        return self.hardware_info


# Global instance
_hardware_optimizer = None


def get_hardware_optimizer() -> HardwareOptimizationManager:
    """Get the global hardware optimization manager instance."""
    global _hardware_optimizer
    if _hardware_optimizer is None:
        _hardware_optimizer = HardwareOptimizationManager()
    return _hardware_optimizer
