"""
Statistics tracking for TinyIMGserver.

This module provides functionality to track server statistics including
generation requests, queue status, and performance metrics.
"""

import time
from typing import Dict, Any
from threading import Lock

class ServerStats:
    """Thread-safe server statistics tracker."""
    
    def __init__(self):
        """Initialize statistics tracker."""
        self._lock = Lock()
        self._stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "queue_length": 0,
            "start_time": time.time(),
            "last_generation": None,
            "active_generations": 0
        }
    
    def increment_request(self):
        """Increment total request counter."""
        with self._lock:
            self._stats["total_requests"] += 1
    
    def increment_success(self):
        """Increment successful generation counter."""
        with self._lock:
            self._stats["successful_generations"] += 1
            self._stats["last_generation"] = time.time()
    
    def increment_failure(self):
        """Increment failed generation counter."""
        with self._lock:
            self._stats["failed_generations"] += 1
    
    def set_queue_length(self, length: int):
        """Set current queue length."""
        with self._lock:
            self._stats["queue_length"] = length
    
    def set_active_generations(self, count: int):
        """Set number of active generations."""
        with self._lock:
            self._stats["active_generations"] = count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics snapshot."""
        with self._lock:
            stats = self._stats.copy()
            stats["uptime"] = time.time() - stats["start_time"]
            return stats

# Global statistics instance
server_stats = ServerStats()
