"""
Rate limiting utilities for API calls.

Provides token bucket rate limiting and per-provider concurrency management
to respect API rate limits during parallel experiment execution.
"""

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter for API calls.
    
    Ensures requests are spaced out to respect API rate limits.
    Uses a simple time-based approach where we track the last request
    and ensure minimum intervals between requests.
    """
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self.lock = asyncio.Lock()
        self.last_request = 0.0
    
    async def acquire(self):
        """Wait until it's safe to make another request."""
        async with self.lock:
            now = time.time()
            wait_time = max(0, self.last_request + self.min_interval - now)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request = time.time()


class ProviderRateLimiters:
    """Container for per-provider rate limiters and semaphores.
    
    Manages both:
    1. Concurrent trial limits per provider (semaphores)
    2. Request rate limits per provider (rate limiters)
    """
    def __init__(
        self,
        anthropic_concurrent: int = 10,
        openai_concurrent: int = 10,
        anthropic_rps: float = 10.0,  # requests per second
        openai_rps: float = 20.0,
    ):
        # Semaphores for concurrent trial limits
        self.semaphores = {
            "anthropic": asyncio.Semaphore(anthropic_concurrent),
            "openai": asyncio.Semaphore(openai_concurrent),
            "unknown": asyncio.Semaphore(max(anthropic_concurrent, openai_concurrent)),
        }
        # Rate limiters for API request spacing
        self.rate_limiters = {
            "anthropic": RateLimiter(anthropic_rps),
            "openai": RateLimiter(openai_rps),
            "unknown": RateLimiter(min(anthropic_rps, openai_rps)),
        }
    
    def get_semaphore(self, provider: str) -> asyncio.Semaphore:
        """Get the semaphore for a provider."""
        return self.semaphores.get(provider, self.semaphores["unknown"])
    
    def get_rate_limiter(self, provider: str) -> "RateLimiter":
        """Get the rate limiter for a provider."""
        return self.rate_limiters.get(provider, self.rate_limiters["unknown"])

