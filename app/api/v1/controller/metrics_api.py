"""
API endpoints for performance metrics and monitoring
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from app.common.monitoring.performance import perf_monitor, get_performance_summary
from app.common.cache.memory_cache import memory_cache, query_cache, embedding_cache

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get system performance metrics"""
    try:
        summary = await get_performance_summary()
        return {"status": "success", "data": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache")
async def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics"""
    try:
        memory_stats = await memory_cache.get_stats()
        query_stats = await query_cache.get_stats()
        embedding_stats = await embedding_cache.get_stats()

        return {
            "status": "success",
            "data": {
                "memory_cache": memory_stats,
                "query_cache": query_stats,
                "embedding_cache": embedding_stats,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_metrics() -> Dict[str, str]:
    """Reset all performance metrics"""
    try:
        await perf_monitor.reset_metrics()
        return {
            "status": "success",
            "message": "Performance metrics reset successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check with performance indicators"""
    try:
        # Get basic metrics
        metrics = await perf_monitor.get_metrics()

        # Calculate health score based on performance
        total_ops = sum(m["count"] for m in metrics.values())
        avg_success_rate = (
            sum(m["success_count"] for m in metrics.values()) / total_ops
            if total_ops > 0
            else 1.0
        )

        # Get cache health
        memory_stats = await memory_cache.get_stats()
        cache_hit_rate = memory_stats["hit_count"] / (
            memory_stats["total_items"] + 1
        )  # Avoid division by zero

        health_score = (avg_success_rate * 0.7) + (cache_hit_rate * 0.3)

        return {
            "status": "healthy" if health_score > 0.8 else "degraded",
            "health_score": round(health_score * 100, 2),
            "metrics": {
                "total_operations": total_ops,
                "success_rate": round(avg_success_rate * 100, 2),
                "cache_hit_rate": round(cache_hit_rate * 100, 2),
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
