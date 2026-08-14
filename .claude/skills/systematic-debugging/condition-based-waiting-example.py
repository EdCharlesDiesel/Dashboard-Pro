"""
Condition-based waiting utilities for async tests.

Ported from: Lace test infrastructure improvements (2025-10-03)
Context: Fixed 15 flaky tests by replacing arbitrary timeouts/sleeps
with polling for the actual condition you care about.

These assume an async ThreadManager (get_events is a coroutine). If your
ThreadManager is synchronous, drop the `await` on `thread_manager.get_events(...)`
calls below.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

LaceEventType = str  # narrow this to a Literal[...] / Enum if you have a fixed set


@dataclass
class LaceEvent:
    type: LaceEventType
    data: Any = None


@runtime_checkable
class ThreadManager(Protocol):
    async def get_events(self, thread_id: str) -> list[LaceEvent]: ...


POLL_INTERVAL_S = 0.01  # 10ms, matches the original poll rate
DEFAULT_TIMEOUT_S = 5.0


async def wait_for_event(
        thread_manager: ThreadManager,
        thread_id: str,
        event_type: LaceEventType,
        timeout_s: float = DEFAULT_TIMEOUT_S,
) -> LaceEvent:
    """
    Wait for a specific event type to appear in a thread.

    Example:
        await wait_for_event(thread_manager, agent_thread_id, "TOOL_RESULT")
    """
    start = time.monotonic()

    while True:
        events = await thread_manager.get_events(thread_id)
        for event in events:
            if event.type == event_type:
                return event

        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"Timeout waiting for {event_type} event after {timeout_s}s"
            )

        await asyncio.sleep(POLL_INTERVAL_S)


async def wait_for_event_count(
        thread_manager: ThreadManager,
        thread_id: str,
        event_type: LaceEventType,
        count: int,
        timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[LaceEvent]:
    """
    Wait for a specific number of events of a given type.

    Example:
        # Wait for 2 AGENT_MESSAGE events (initial response + continuation)
        await wait_for_event_count(thread_manager, agent_thread_id, "AGENT_MESSAGE", 2)
    """
    start = time.monotonic()

    while True:
        events = await thread_manager.get_events(thread_id)
        matching = [e for e in events if e.type == event_type]

        if len(matching) >= count:
            return matching

        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"Timeout waiting for {count} {event_type} events after "
                f"{timeout_s}s (got {len(matching)})"
            )

        await asyncio.sleep(POLL_INTERVAL_S)


async def wait_for_event_match(
        thread_manager: ThreadManager,
        thread_id: str,
        predicate: Callable[[LaceEvent], bool],
        description: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
) -> LaceEvent:
    """
    Wait for an event matching a custom predicate.
    Useful when you need to check event data, not just type.

    Example:
        # Wait for TOOL_RESULT with specific id
        await wait_for_event_match(
            thread_manager,
            agent_thread_id,
            lambda e: e.type == "TOOL_RESULT" and e.data.get("id") == "call_123",
            'TOOL_RESULT with id=call_123'
        )
    """
    start = time.monotonic()

    while True:
        events = await thread_manager.get_events(thread_id)
        for event in events:
            if predicate(event):
                return event

        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"Timeout waiting for {description} after {timeout_s}s"
            )

        await asyncio.sleep(POLL_INTERVAL_S)


# Usage example from actual debugging session:
#
# BEFORE (flaky):
# ---------------
# message_task = asyncio.create_task(agent.send_message("Execute tools"))
# await asyncio.sleep(0.3)   # Hope tools start in 300ms
# agent.abort()
# await message_task
# await asyncio.sleep(0.05)  # Hope results arrive in 50ms
# assert len(tool_results) == 2  # Fails randomly
#
# AFTER (reliable):
# ----------------
# message_task = asyncio.create_task(agent.send_message("Execute tools"))
# await wait_for_event_count(thread_manager, thread_id, "TOOL_CALL", 2)  # Wait for tools to start
# agent.abort()
# await message_task
# await wait_for_event_count(thread_manager, thread_id, "TOOL_RESULT", 2)  # Wait for results
# assert len(tool_results) == 2  # Always succeeds
#
# Result: 60% pass rate -> 100%, 40% faster execution