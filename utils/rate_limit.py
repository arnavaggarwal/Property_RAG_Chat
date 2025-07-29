from datetime import datetime, timedelta
import streamlit as st


class TokenBucket:
    """
    A simple token bucket rate limiter using Streamlit session_state for per-session storage.

    Attributes:
        capacity (int): Maximum number of tokens in the bucket.
        refill_interval (timedelta): Interval at which the bucket is fully refilled.
    """
    def __init__(self, capacity: int, refill_interval_minutes: int):
        self.capacity = capacity
        self.refill_interval = timedelta(minutes=refill_interval_minutes)

    def initialize(self):
        """
        Initialize tokens and last_refill timestamp in session_state if not already present.
        """
        if 'tokens' not in st.session_state:
            st.session_state.tokens = self.capacity
            st.session_state.last_refill = datetime.now()

    def refill(self):
        """
        Refill the token bucket to full capacity if the refill interval has passed.
        """
        now = datetime.now()
        last = st.session_state.last_refill
        if now - last >= self.refill_interval:
            st.session_state.tokens = self.capacity
            st.session_state.last_refill = now

    def consume(self, amount: int = 1) -> bool:
        """
        Attempt to consume a given number of tokens.

        Returns:
            bool: True if tokens were consumed; False if not enough tokens available.
        """
        # Ensure tokens are up to date before consuming
        self.refill()
        if st.session_state.tokens >= amount:
            st.session_state.tokens -= amount
            return True
        return False

    def remaining(self) -> int:
        """
        Returns the current number of tokens available, after refilling if needed.
        """
        self.refill()
        return st.session_state.tokens

    def retry_after_minutes(self) -> int:
        """
        Calculate how many minutes remain until the next full refill.

        Returns:
            int: Minutes until bucket is next refilled.
        """
        now = datetime.now()
        next_refill = st.session_state.last_refill + self.refill_interval
        delta = next_refill - now
        # Round up to the next full minute
        return max(int(delta.total_seconds() // 60) + 1, 0)
