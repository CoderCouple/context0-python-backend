"""
Performance monitoring and profiling utilities
"""
import asyncio
import time
import functools
import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Track and log performance metrics"""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def record_metric(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a performance metric"""

        async with self._lock:
            if operation not in self.metrics:
                self.metrics[operation] = {
                    "count": 0,
                    "total_duration_ms": 0,
                    "min_duration_ms": float("inf"),
                    "max_duration_ms": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "last_recorded": None,
                }

            metric = self.metrics[operation]
            metric["count"] += 1
            metric["total_duration_ms"] += duration_ms
            metric["min_duration_ms"] = min(metric["min_duration_ms"], duration_ms)
            metric["max_duration_ms"] = max(metric["max_duration_ms"], duration_ms)

            if success:
                metric["success_count"] += 1
            else:
                metric["failure_count"] += 1

            metric["last_recorded"] = datetime.utcnow()

            # Log slow operations
            if duration_ms > 1000:  # Log operations slower than 1 second
                logger.warning(f"Slow operation: {operation} took {duration_ms:.2f}ms")

    async def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics with calculated averages"""

        async with self._lock:
            result = {}
            for operation, data in self.metrics.items():
                avg_duration = (
                    data["total_duration_ms"] / data["count"]
                    if data["count"] > 0
                    else 0
                )
                success_rate = (
                    data["success_count"] / data["count"] if data["count"] > 0 else 0
                )

                result[operation] = {
                    **data,
                    "avg_duration_ms": avg_duration,
                    "success_rate": success_rate,
                }

            return result

    async def reset_metrics(self):
        """Reset all metrics"""
        async with self._lock:
            self.metrics.clear()


# Global performance monitor instance
perf_monitor = PerformanceMonitor()


def track_performance(operation_name: Optional[str] = None):
    """Decorator to track async function performance"""

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            success = True

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                await perf_monitor.record_metric(name, duration_ms, success)

        return wrapper

    return decorator


@asynccontextmanager
async def measure_time(operation_name: str):
    """Context manager to measure execution time"""

    start_time = time.time()
    success = True

    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        await perf_monitor.record_metric(operation_name, duration_ms, success)


class RequestProfiler:
    """Profile complete request lifecycle"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.time()
        self.checkpoints: List[Dict[str, Any]] = []

    def checkpoint(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a checkpoint in request processing"""

        elapsed_ms = (time.time() - self.start_time) * 1000
        self.checkpoints.append(
            {
                "name": name,
                "elapsed_ms": elapsed_ms,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {},
            }
        )

    def get_profile(self) -> Dict[str, Any]:
        """Get complete request profile"""

        total_duration_ms = (time.time() - self.start_time) * 1000

        # Calculate durations between checkpoints
        checkpoint_durations = []
        for i in range(len(self.checkpoints)):
            if i == 0:
                duration = self.checkpoints[i]["elapsed_ms"]
            else:
                duration = (
                    self.checkpoints[i]["elapsed_ms"]
                    - self.checkpoints[i - 1]["elapsed_ms"]
                )

            checkpoint_durations.append(
                {
                    "name": self.checkpoints[i]["name"],
                    "duration_ms": duration,
                    "cumulative_ms": self.checkpoints[i]["elapsed_ms"],
                }
            )

        return {
            "request_id": self.request_id,
            "total_duration_ms": total_duration_ms,
            "checkpoints": checkpoint_durations,
            "slowest_operation": max(
                checkpoint_durations, key=lambda x: x["duration_ms"]
            )
            if checkpoint_durations
            else None,
        }


# Middleware for automatic request profiling
class PerformanceMiddleware:
    """FastAPI middleware for performance tracking"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            method = scope["method"]

            start_time = time.time()

            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    duration_ms = (time.time() - start_time) * 1000

                    # Track API endpoint performance
                    await perf_monitor.record_metric(
                        f"api.{method}.{path}",
                        duration_ms,
                        success=message.get("status", 200) < 400,
                    )

                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Utility functions for performance analysis


async def analyze_slow_queries(db_collection, threshold_ms: float = 100):
    """Analyze slow database queries"""

    # Enable profiling
    await db_collection.database.command("profile", 2, slowms=int(threshold_ms))

    # Get profiling data
    profile_collection = db_collection.database.system.profile
    slow_queries = (
        await profile_collection.find({"millis": {"$gte": threshold_ms}})
        .sort("millis", -1)
        .limit(10)
        .to_list(None)
    )

    analysis = []
    for query in slow_queries:
        analysis.append(
            {
                "operation": query.get("op"),
                "namespace": query.get("ns"),
                "duration_ms": query.get("millis"),
                "timestamp": query.get("ts"),
                "command": query.get("command", {}),
            }
        )

    return analysis


async def get_performance_summary() -> Dict[str, Any]:
    """Get overall system performance summary"""

    metrics = await perf_monitor.get_metrics()

    # Calculate overall statistics
    total_operations = sum(m["count"] for m in metrics.values())
    total_duration_ms = sum(m["total_duration_ms"] for m in metrics.values())
    overall_success_rate = (
        sum(m["success_count"] for m in metrics.values()) / total_operations
        if total_operations > 0
        else 0
    )

    # Find slowest operations
    slowest_ops = sorted(
        [(op, data["avg_duration_ms"]) for op, data in metrics.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    # Find most frequent operations
    most_frequent = sorted(
        [(op, data["count"]) for op, data in metrics.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "total_operations": total_operations,
        "total_duration_ms": total_duration_ms,
        "overall_success_rate": overall_success_rate,
        "slowest_operations": [
            {"operation": op, "avg_duration_ms": duration}
            for op, duration in slowest_ops
        ],
        "most_frequent_operations": [
            {"operation": op, "count": count} for op, count in most_frequent
        ],
        "detailed_metrics": metrics,
    }
