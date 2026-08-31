import os
import time
from typing import Dict, Any

class MetricsService:
    """
    Tracking service for system metrics:
    - Total queries processed
    - Token consumption (prompt_tokens, completion_tokens, total_tokens) directly from response.usage
    - Monthly token quota tracking and usage percentage
    - Accumulated API cost calculation (USD)
    - Escalation count & escalation rate (%)
    - Latency statistics
    """

    # Model token pricing per 1,000 tokens (USD)
    PRICING_PER_1K = {
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
        "gpt-4o": {"prompt": 0.00250, "completion": 0.01000},
        "gpt-3.5-turbo": {"prompt": 0.00150, "completion": 0.00200},
        "gemini-3.5-flash-lite": {"prompt": 0.000075, "completion": 0.00030},
        "gemini-3.5-flash": {"prompt": 0.000075, "completion": 0.00030},
        "gemini-3.6-flash": {"prompt": 0.000075, "completion": 0.00030},
        "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.00002},
        "gemini-embedding-001": {"prompt": 0.00002, "completion": 0.00002},
        "default": {"prompt": 0.00015, "completion": 0.00060}
    }

    def __init__(self, token_limit_monthly: int = None):
        self.start_time = time.time()
        self.total_queries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.total_escalations = 0
        self.total_pre_llm_escalations = 0
        self.total_post_llm_escalations = 0
        
        # Monthly token quota limit (default 1,000,000 tokens)
        self.token_limit_monthly = token_limit_monthly or int(os.getenv("TOKEN_LIMIT_MONTHLY", 1_000_000))

    def record_usage(self, prompt_tokens: int, completion_tokens: int, model_name: str = "gpt-4o-mini") -> float:
        """
        Calculates and records cost based on exact response.usage object from OpenAI API call.
        """
        pricing = self.PRICING_PER_1K.get(model_name, self.PRICING_PER_1K["default"])
        
        prompt_cost = (prompt_tokens / 1000.0) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000.0) * pricing["completion"]
        query_cost = prompt_cost + completion_cost

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.total_cost_usd += query_cost

        return round(query_cost, 6)

    def record_query(self, escalated: bool = False, pre_llm: bool = False):
        self.total_queries += 1
        if escalated:
            self.total_escalations += 1
            if pre_llm:
                self.total_pre_llm_escalations += 1
            else:
                self.total_post_llm_escalations += 1

    def get_metrics(self, cache_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        uptime_seconds = int(time.time() - self.start_time)
        escalation_rate = (self.total_escalations / self.total_queries * 100.0) if self.total_queries > 0 else 0.0
        usage_pct = (self.total_tokens / self.token_limit_monthly * 100.0) if self.token_limit_monthly > 0 else 0.0

        metrics = {
            "total_queries_processed": self.total_queries,
            "total_tokens_used": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "token_limit_monthly": self.token_limit_monthly,
            "token_usage_percentage": round(usage_pct, 4),
            "escalation_metrics": {
                "total_escalations": self.total_escalations,
                "escalation_rate_pct": round(escalation_rate, 2),
                "pre_llm_escalations_saved_cost": self.total_pre_llm_escalations,
                "post_llm_escalations": self.total_post_llm_escalations
            },
            "token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "token_limit_monthly": self.token_limit_monthly,
                "token_usage_percentage": round(usage_pct, 4)
            },
            "financial_metrics": {
                "total_estimated_cost_usd": round(self.total_cost_usd, 6),
                "currency": "USD"
            },
            "system_uptime_seconds": uptime_seconds
        }

        if cache_stats:
            metrics["cache_metrics"] = cache_stats

        return metrics
