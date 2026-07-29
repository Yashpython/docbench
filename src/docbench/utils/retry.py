from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

T = TypeVar("T")


def with_retries(fn: Callable[[], T], attempts: int = 3) -> T:
    """Retries transient failures (rate limits, timeouts) with exponential backoff.
    Not yet wired into the adapters below -- each currently catches its single API
    call and degrades to AdapterResult(error=...) on any exception, which is safe
    but doesn't distinguish "call worth retrying" from "call worth recording as a
    failure". Wrap the provider .create()/.generate_content() call in this at the
    point adapters start seeing real rate-limit noise in a run.
    """
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )(fn)()
