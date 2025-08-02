"""
Memory profiling utilities for tracking memory usage throughout the pipeline.
"""
import psutil
import os
import gc
import tracemalloc
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime


class MemoryProfiler:
    """Track and report memory usage at different stages of the pipeline."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.snapshots: List[Tuple[str, float, Optional[tracemalloc.Snapshot]]] = []
        self.use_tracemalloc = False
        
    def start_detailed_tracking(self):
        """Start detailed memory tracking with tracemalloc."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.use_tracemalloc = True
    
    def stop_detailed_tracking(self):
        """Stop detailed memory tracking."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            self.use_tracemalloc = False
    
    def snapshot(self, label: str):
        """Take a memory snapshot with the given label."""
        # Force garbage collection before measuring
        gc.collect()
        
        # Get current memory usage in MB
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        # Get tracemalloc snapshot if enabled
        trace_snapshot = None
        if self.use_tracemalloc and tracemalloc.is_tracing():
            trace_snapshot = tracemalloc.take_snapshot()
        
        self.snapshots.append((label, memory_mb, trace_snapshot))
        
        print(f"\n[Memory] {label}: {memory_mb:.2f} MB")
        
        # Show memory increase from last snapshot
        if len(self.snapshots) > 1:
            prev_label, prev_memory, _ = self.snapshots[-2]
            diff = memory_mb - prev_memory
            print(f"[Memory] Increase from '{prev_label}': {diff:+.2f} MB")
    
    def report(self):
        """Generate a detailed memory usage report."""
        print("\n" + "="*60)
        print("MEMORY USAGE REPORT")
        print("="*60)
        
        for i, (label, memory_mb, snapshot) in enumerate(self.snapshots):
            print(f"\n{i+1}. {label}: {memory_mb:.2f} MB")
            
            if i > 0:
                prev_memory = self.snapshots[i-1][1]
                diff = memory_mb - prev_memory
                print(f"   Change: {diff:+.2f} MB")
        
        # Show total memory increase
        if len(self.snapshots) >= 2:
            total_increase = self.snapshots[-1][1] - self.snapshots[0][1]
            print(f"\nTotal memory increase: {total_increase:.2f} MB")
            print(f"Peak memory usage: {max(s[1] for s in self.snapshots):.2f} MB")
    
    def get_object_sizes(self, objects: Dict[str, object]) -> Dict[str, float]:
        """Get the size of Python objects in MB."""
        sizes = {}
        for name, obj in objects.items():
            if isinstance(obj, np.ndarray):
                size_mb = obj.nbytes / 1024 / 1024
            elif hasattr(obj, '__sizeof__'):
                size_mb = obj.__sizeof__() / 1024 / 1024
            else:
                size_mb = 0
            sizes[name] = size_mb
        return sizes
    
    def print_large_objects(self, objects: Dict[str, object], threshold_mb: float = 10.0):
        """Print objects larger than threshold."""
        sizes = self.get_object_sizes(objects)
        large_objects = {k: v for k, v in sizes.items() if v > threshold_mb}
        
        if large_objects:
            print(f"\n[Memory] Objects larger than {threshold_mb} MB:")
            for name, size in sorted(large_objects.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {name}: {size:.2f} MB")
    
    @staticmethod
    def clear_memory(objects_to_delete: List[str], local_vars: dict):
        """Clear specified objects from memory."""
        for obj_name in objects_to_delete:
            if obj_name in local_vars:
                del local_vars[obj_name]
        gc.collect()
        print(f"[Memory] Cleared {len(objects_to_delete)} objects from memory")
