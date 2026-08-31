import time
import hashlib
from typing import Dict, Any, Optional

class CacheService:
    """
    In-memory semantic and exact matching cache for query responses.
    Reduces latency, API costs, and server load for repeated admissions queries.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def _normalize_query(self, query: str) -> str:
        """Normalizes text by lowercasing, stripping punctuation and extra whitespace."""
        cleaned = "".join(c.lower() for c in query if c.isalnum() or c.isspace())
        return " ".join(cleaned.split())

    def _get_key(self, query: str) -> str:
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._get_key(query)
        entry = self.cache.get(key)
        if entry:
            # Check TTL
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                self.hits += 1
                cached_data = entry["data"].copy()
                cached_data["cached"] = True
                return cached_data
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None

    def set(self, query: str, data: Dict[str, Any]):
        key = self._get_key(query)
        self.cache[key] = {
            "timestamp": time.time(),
            "data": data
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": round(hit_rate, 2),
            "cached_entries_count": len(self.cache)
        }

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
