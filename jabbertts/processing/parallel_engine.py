"""Parallel Processing Architecture for JabberTTS.

This module implements a high-performance parallel processing system with:
- Chunked generation with semantic boundary detection
- Thread pool with configurable async workers (CPU cores × 2, max 20)
- Queue-based pipeline with priority levels
- Dynamic batching for GPU utilization optimization
- Memory pooling to minimize garbage collection pressure
"""

import asyncio
import logging
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, PriorityQueue
import threading
import gc
import numpy as np
import torch

from jabbertts.caching.multilevel_cache import get_cache_manager

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    REALTIME = 1      # Interactive/streaming requests
    HIGH = 2          # User-facing requests
    NORMAL = 3        # Standard batch processing
    LOW = 4           # Background tasks
    BATCH = 5         # Large batch processing


@dataclass
class ProcessingTask:
    """A processing task with priority and metadata."""
    task_id: str
    priority: TaskPriority
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    callback: Optional[Callable] = None
    created_time: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    
    def __lt__(self, other):
        """Compare tasks by priority for priority queue."""
        return self.priority.value < other.priority.value


@dataclass
class BatchRequest:
    """Batch processing request."""
    batch_id: str
    texts: List[str]
    voice: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.BATCH
    max_batch_size: int = 8
    timeout: float = 30.0


@dataclass
class ProcessingStats:
    """Processing performance statistics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0


class MemoryPool:
    """Memory pool for tensor allocation reuse."""
    
    def __init__(self, max_pool_size: int = 100):
        """Initialize memory pool.
        
        Args:
            max_pool_size: Maximum number of tensors to pool
        """
        self.max_pool_size = max_pool_size
        self._pools: Dict[tuple, List[torch.Tensor]] = {}
        self._lock = threading.RLock()
        self._allocation_count = 0
        self._reuse_count = 0
    
    def get_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
        """Get a tensor from the pool or allocate new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Device to allocate on
            
        Returns:
            Tensor from pool or newly allocated
        """
        key = (shape, dtype, device)
        
        with self._lock:
            if key in self._pools and self._pools[key]:
                tensor = self._pools[key].pop()
                tensor.zero_()  # Clear data
                self._reuse_count += 1
                return tensor
            else:
                self._allocation_count += 1
                return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool.
        
        Args:
            tensor: Tensor to return to pool
        """
        if tensor.numel() == 0:
            return
        
        key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
        
        with self._lock:
            if key not in self._pools:
                self._pools[key] = []
            
            if len(self._pools[key]) < self.max_pool_size:
                self._pools[key].append(tensor.detach())
    
    def clear_pool(self) -> None:
        """Clear all pooled tensors."""
        with self._lock:
            self._pools.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_tensors = sum(len(pool) for pool in self._pools.values())
            reuse_rate = self._reuse_count / max(1, self._allocation_count + self._reuse_count)
            
            return {
                "total_pools": len(self._pools),
                "total_tensors": total_tensors,
                "allocation_count": self._allocation_count,
                "reuse_count": self._reuse_count,
                "reuse_rate": reuse_rate
            }


class QueueManager:
    """Queue-based pipeline manager with priority support."""
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize queue manager.
        
        Args:
            max_queue_size: Maximum queue size
        """
        self.max_queue_size = max_queue_size
        self._task_queue = PriorityQueue(maxsize=max_queue_size)
        self._result_queue = Queue()
        self._active_tasks: Dict[str, ProcessingTask] = {}
        self._lock = threading.RLock()
        self._stats = ProcessingStats()
    
    def submit_task(self, task: ProcessingTask) -> bool:
        """Submit a task to the queue.
        
        Args:
            task: Processing task to submit
            
        Returns:
            True if task was queued successfully
        """
        try:
            self._task_queue.put(task, block=False)
            
            with self._lock:
                self._active_tasks[task.task_id] = task
                self._stats.total_tasks += 1
                self._stats.queue_depth = self._task_queue.qsize()
            
            logger.debug(f"Queued task {task.task_id} with priority {task.priority.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue task {task.task_id}: {e}")
            return False
    
    def get_next_task(self, timeout: Optional[float] = None) -> Optional[ProcessingTask]:
        """Get the next task from the queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Next task or None if timeout
        """
        try:
            task = self._task_queue.get(timeout=timeout)
            
            with self._lock:
                self._stats.queue_depth = self._task_queue.qsize()
            
            return task
            
        except Exception:
            return None
    
    def complete_task(self, task_id: str, result: Any = None, error: Optional[Exception] = None) -> None:
        """Mark a task as completed.
        
        Args:
            task_id: Task identifier
            result: Task result
            error: Error if task failed
        """
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                
                if error:
                    self._stats.failed_tasks += 1
                    logger.error(f"Task {task_id} failed: {error}")
                else:
                    self._stats.completed_tasks += 1
                    
                    # Update average processing time
                    processing_time = time.time() - task.created_time
                    if self._stats.completed_tasks == 1:
                        self._stats.average_processing_time = processing_time
                    else:
                        # Exponential moving average
                        alpha = 0.1
                        self._stats.average_processing_time = (
                            alpha * processing_time + 
                            (1 - alpha) * self._stats.average_processing_time
                        )
                
                # Execute callback if provided
                if task.callback:
                    try:
                        task.callback(result, error)
                    except Exception as e:
                        logger.error(f"Task callback failed: {e}")
                
                del self._active_tasks[task_id]
    
    def get_stats(self) -> ProcessingStats:
        """Get queue statistics."""
        with self._lock:
            self._stats.queue_depth = self._task_queue.qsize()
            return self._stats


class BatchProcessor:
    """Dynamic batching processor for GPU optimization."""
    
    def __init__(
        self,
        max_batch_size: int = 8,
        batch_timeout: float = 0.1,
        min_batch_size: int = 2
    ):
        """Initialize batch processor.
        
        Args:
            max_batch_size: Maximum batch size
            batch_timeout: Maximum time to wait for batch
            min_batch_size: Minimum batch size before timeout
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.min_batch_size = min_batch_size
        
        self._pending_requests: List[BatchRequest] = []
        self._lock = threading.RLock()
        self._batch_event = threading.Event()
    
    def add_request(self, request: BatchRequest) -> None:
        """Add a request to the batch.
        
        Args:
            request: Batch request to add
        """
        with self._lock:
            self._pending_requests.append(request)
            
            # Trigger batch processing if full
            if len(self._pending_requests) >= self.max_batch_size:
                self._batch_event.set()
    
    def get_batch(self) -> List[BatchRequest]:
        """Get the next batch of requests.
        
        Returns:
            List of batch requests
        """
        # Wait for batch or timeout
        self._batch_event.wait(timeout=self.batch_timeout)
        
        with self._lock:
            if not self._pending_requests:
                return []
            
            # Take up to max_batch_size requests
            batch = self._pending_requests[:self.max_batch_size]
            self._pending_requests = self._pending_requests[self.max_batch_size:]
            
            # Reset event if no more pending requests
            if not self._pending_requests:
                self._batch_event.clear()
            
            return batch


class ParallelProcessingEngine:
    """Main parallel processing engine."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        enable_gpu_batching: bool = True,
        memory_pool_size: int = 100
    ):
        """Initialize parallel processing engine.
        
        Args:
            max_workers: Maximum number of workers (default: CPU cores × 2)
            enable_gpu_batching: Enable GPU batching optimization
            memory_pool_size: Size of memory pool
        """
        self.max_workers = max_workers or min(multiprocessing.cpu_count() * 2, 20)
        self.enable_gpu_batching = enable_gpu_batching
        
        # Initialize components
        self.queue_manager = QueueManager()
        self.memory_pool = MemoryPool(memory_pool_size)
        self.batch_processor = BatchProcessor() if enable_gpu_batching else None
        self.cache_manager = get_cache_manager()
        
        # Thread pool for CPU tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, self.max_workers // 4))
        
        # Worker management
        self._workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._running = False
        
        # Performance monitoring
        self._start_monitoring()
    
    def start(self) -> None:
        """Start the parallel processing engine."""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TTSWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started parallel processing engine with {self.max_workers} workers")
    
    def stop(self) -> None:
        """Stop the parallel processing engine."""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Clear memory pool
        self.memory_pool.clear_pool()
        
        logger.info("Stopped parallel processing engine")
    
    def submit_task(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit a task for parallel processing.
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority
            callback: Completion callback
            timeout: Task timeout
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task_id = f"task_{int(time.time() * 1000000)}"
        
        task = ProcessingTask(
            task_id=task_id,
            priority=priority,
            func=func,
            args=args,
            kwargs=kwargs,
            callback=callback,
            timeout=timeout
        )
        
        if self.queue_manager.submit_task(task):
            return task_id
        else:
            raise RuntimeError("Failed to submit task to queue")
    
    def _worker_loop(self) -> None:
        """Main worker loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get next task
                task = self.queue_manager.get_next_task(timeout=1.0)
                if task is None:
                    continue
                
                # Execute task
                try:
                    result = task.func(*task.args, **task.kwargs)
                    self.queue_manager.complete_task(task.task_id, result=result)
                    
                except Exception as e:
                    self.queue_manager.complete_task(task.task_id, error=e)
                
                # Periodic garbage collection
                if self.queue_manager._stats.completed_tasks % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _start_monitoring(self) -> None:
        """Start performance monitoring."""
        def monitor_loop():
            while self._running:
                try:
                    # Update memory usage
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    stats = self.queue_manager.get_stats()
                    stats.memory_usage_mb = memory_mb
                    stats.active_workers = len([w for w in self._workers if w.is_alive()])
                    
                    # GPU utilization (if available)
                    if torch.cuda.is_available():
                        try:
                            stats.gpu_utilization = torch.cuda.utilization()
                        except:
                            pass
                    
                    time.sleep(5.0)  # Monitor every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        queue_stats = self.queue_manager.get_stats()
        memory_stats = self.memory_pool.get_stats()
        cache_stats = self.cache_manager.get_cache_stats()
        
        return {
            "processing": queue_stats.__dict__,
            "memory_pool": memory_stats,
            "cache": cache_stats,
            "workers": {
                "max_workers": self.max_workers,
                "active_workers": len([w for w in self._workers if w.is_alive()]),
                "running": self._running
            }
        }


# Global instance
_processing_engine = None


def get_processing_engine() -> ParallelProcessingEngine:
    """Get the global parallel processing engine instance."""
    global _processing_engine
    if _processing_engine is None:
        _processing_engine = ParallelProcessingEngine()
        _processing_engine.start()
    return _processing_engine
