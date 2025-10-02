# rate_limiter.py
"""Rate limiting and API error handling module with exponential backoff.

Provides centralized control for API requests to prevent rate limit errors
and token overflow issues.
"""

import time
import re
from typing import Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RateLimitState:
    """State tracking for rate limiting."""
    last_request_time: float = 0.0
    requests_in_window: int = 0
    window_start: float = 0.0
    retry_after: Optional[float] = None
    consecutive_errors: int = 0


class RateLimiter:
    """Centralized rate limiter with exponential backoff and error handling.
    
    Handles:
    - RPM (requests per minute) limits
    - Token limits with estimation
    - API error detection and automatic retry
    - Exponential backoff on repeated errors
    """
    
    def __init__(self, rpm_limit: int = 10, max_tokens_per_request: int = 8000):
        """Initialize rate limiter.
        
        Args:
            rpm_limit: Maximum requests per minute
            max_tokens_per_request: Estimated max tokens per request
        """
        self.rpm_limit = rpm_limit
        self.max_tokens_per_request = max_tokens_per_request
        self.state = RateLimitState()
        self.min_interval = 60.0 / rpm_limit  # seconds between requests
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation).
        
        Uses ~4 chars per token as a conservative estimate.
        """
        return len(text) // 4
    
    def _extract_retry_after(self, error_message: str) -> Optional[float]:
        """Extract retry-after duration from error message.
        
        Looks for patterns like:
        - "retry after 30s"
        - "try again in 1m"
        - "Rate limit exceeded. Please retry after 45 seconds"
        """
        patterns = [
            r'retry after (\d+)s',
            r'retry after (\d+) second',
            r'try again in (\d+)m',
            r'try again in (\d+) minute',
            r'wait (\d+) second',
            r'Please retry after (\d+) second',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                # Convert minutes to seconds if needed
                if 'minute' in pattern or pattern.endswith('m'):
                    return value * 60.0
                return float(value)
        
        return None
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit',
            'too many requests',
            'quota exceeded',
            '429',
            'requests per minute',
            'rpm',
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def _is_token_limit_error(self, error: Exception) -> bool:
        """Check if error is a token limit error."""
        error_str = str(error).lower()
        token_limit_indicators = [
            'token limit',
            'context length',
            'maximum context',
            'too many tokens',
            'tokens exceeded',
            'context_length_exceeded',
        ]
        return any(indicator in error_str for indicator in token_limit_indicators)
    
    def _calculate_backoff(self) -> float:
        """Calculate exponential backoff duration based on consecutive errors.
        
        Returns wait time in seconds: 2^errors seconds (capped at 60s).
        """
        if self.state.consecutive_errors == 0:
            return 0.0
        
        backoff = min(2 ** self.state.consecutive_errors, 60.0)
        return backoff
    
    def wait_if_needed(self) -> Optional[dict]:
        """Check if we need to wait and return wait info if so.
        
        Returns:
            None if no wait needed, or dict with wait info:
            {
                'wait_seconds': float,
                'reason': str,
                'retry_at': datetime
            }
        """
        current_time = time.time()
        
        # Check if we're in a retry-after period
        if self.state.retry_after and current_time < self.state.retry_after:
            wait_seconds = self.state.retry_after - current_time
            return {
                'wait_seconds': wait_seconds,
                'reason': 'Rate limit atingido',
                'retry_at': datetime.fromtimestamp(self.state.retry_after)
            }
        
        # Check if we need exponential backoff
        backoff = self._calculate_backoff()
        if backoff > 0:
            elapsed = current_time - self.state.last_request_time
            if elapsed < backoff:
                wait_seconds = backoff - elapsed
                return {
                    'wait_seconds': wait_seconds,
                    'reason': f'Backoff após {self.state.consecutive_errors} erros consecutivos',
                    'retry_at': datetime.fromtimestamp(current_time + wait_seconds)
                }
        
        # Check RPM limit
        elapsed = current_time - self.state.last_request_time
        if elapsed < self.min_interval:
            wait_seconds = self.min_interval - elapsed
            return {
                'wait_seconds': wait_seconds,
                'reason': f'Limite de {self.rpm_limit} requisições/minuto',
                'retry_at': datetime.fromtimestamp(current_time + wait_seconds)
            }
        
        return None
    
    def execute_with_retry(
        self, 
        func: Callable[[], Any], 
        max_retries: int = 3,
        on_wait: Optional[Callable[[dict], None]] = None
    ) -> Any:
        """Execute function with automatic retry on rate limit/token errors.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retries
            on_wait: Optional callback called when waiting (receives wait_info dict)
            
        Returns:
            Result from func()
            
        Raises:
            Exception: If all retries exhausted or non-recoverable error
        """
        attempt = 0
        
        while attempt <= max_retries:
            # Check if we need to wait
            wait_info = self.wait_if_needed()
            if wait_info and on_wait:
                on_wait(wait_info)
            elif wait_info:
                # No callback, just sleep
                time.sleep(wait_info['wait_seconds'])
            
            try:
                # Update state before request
                self.state.last_request_time = time.time()
                
                # Execute the function
                result = func()
                
                # Success - reset error counter
                self.state.consecutive_errors = 0
                self.state.retry_after = None
                
                return result
                
            except Exception as e:
                # Check if it's a recoverable error
                is_rate_limit = self._is_rate_limit_error(e)
                is_token_limit = self._is_token_limit_error(e)
                
                if is_rate_limit or is_token_limit:
                    self.state.consecutive_errors += 1
                    
                    # Extract retry-after if available
                    retry_after_seconds = self._extract_retry_after(str(e))
                    if retry_after_seconds:
                        self.state.retry_after = time.time() + retry_after_seconds
                    
                    if attempt < max_retries:
                        # Calculate wait time
                        if retry_after_seconds:
                            wait_seconds = retry_after_seconds
                        else:
                            wait_seconds = self._calculate_backoff()
                        
                        # Notify about wait
                        if on_wait:
                            wait_info = {
                                'wait_seconds': wait_seconds,
                                'reason': 'Rate limit' if is_rate_limit else 'Token limit',
                                'retry_at': datetime.fromtimestamp(time.time() + wait_seconds),
                                'error': str(e)
                            }
                            on_wait(wait_info)
                        else:
                            time.sleep(wait_seconds)
                        
                        attempt += 1
                        continue
                    else:
                        # Max retries reached
                        raise Exception(
                            f"Máximo de tentativas atingido após {max_retries} tentativas. "
                            f"Último erro: {str(e)}"
                        )
                else:
                    # Non-recoverable error
                    raise
        
        raise Exception("Falha inesperada no loop de retry")


# Global rate limiter instance
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter(rpm_limit: int = 10) -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter(rpm_limit=rpm_limit)
    return _global_limiter


def reset_rate_limiter():
    """Reset global rate limiter (useful for testing)."""
    global _global_limiter
    _global_limiter = None
