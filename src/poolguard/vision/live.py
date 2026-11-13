"""Resilient live frame acquisition, independent of any camera SDK.

A live source is given one thing: a function that opens a stream of BGR
images. Everything else lives here, so every live camera gets the same
behaviour:

- Health is judged by frames arriving, never by connection state. On
  2026-10-12 a LiveKit room stayed "connected" for 21 minutes with no video.
- When frames go quiet, the stream is closed and opened again from scratch,
  with backoff, for as long as the process runs.
- Only the newest frame is kept. A detector slower than the camera should see
  what is happening now, not work through a backlog.

The stream runs on its own thread and event loop; `LiveFrameSource` is the
synchronous `FrameSource` the rest of the pipeline consumes.
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from poolguard.vision.frames import Frame

logger = logging.getLogger(__name__)

OpenStream = Callable[[], AbstractAsyncContextManager[AsyncIterator[np.ndarray]]]
Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]

POLL_SECONDS = 1.0
"""How long `frames()` waits per attempt, so closing the generator is prompt."""
SHUTDOWN_TIMEOUT_SECONDS = 10.0
MAX_BACKOFF_DOUBLINGS = 32


class RejoinPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    stale_after_seconds: float = Field(gt=0)
    backoff_base_seconds: float = Field(gt=0)
    backoff_cap_seconds: float = Field(gt=0)


def backoff_delay(attempt: int, *, base_seconds: float, cap_seconds: float) -> float:
    """Seconds to wait before rejoin number `attempt` (1-based), doubling up to the cap."""
    doublings = min(max(attempt - 1, 0), MAX_BACKOFF_DOUBLINGS)
    return min(cap_seconds, base_seconds * 2**doublings)


class LatestFrameSlot:
    """A one-frame mailbox shared between the stream thread and the consumer."""

    def __init__(self, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._condition = threading.Condition()
        self._image: np.ndarray | None = None
        self._dropped = 0
        self._last_put_at = clock()

    def put(self, image: np.ndarray) -> None:
        with self._condition:
            if self._image is not None:
                self._dropped += 1
            self._image = image
            self._last_put_at = self._clock()
            self._condition.notify()

    def get(self, timeout_seconds: float) -> np.ndarray | None:
        """Take the newest unread frame, waiting up to the timeout for one."""
        with self._condition:
            if self._image is None:
                self._condition.wait(timeout_seconds)
            image = self._image
            self._image = None
            return image

    @property
    def dropped(self) -> int:
        """Frames replaced before anyone read them."""
        with self._condition:
            return self._dropped

    def seconds_since_last_put(self) -> float:
        """Age of the newest frame, or time since creation if none has arrived."""
        with self._condition:
            return self._clock() - self._last_put_at


async def pump(
    open_stream: OpenStream,
    slot: LatestFrameSlot,
    policy: RejoinPolicy,
    *,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Relay frames into the slot forever, reopening the stream whenever it fails.

    Runs until cancelled.
    """
    failures = 0
    while True:
        delivered = 0
        try:
            async with open_stream() as stream:
                iterator = aiter(stream)
                while True:
                    image = await asyncio.wait_for(anext(iterator), policy.stale_after_seconds)
                    slot.put(image)
                    delivered += 1
        except TimeoutError:
            reason = f"no frame for {policy.stale_after_seconds:.0f} s"
        except StopAsyncIteration:
            reason = "stream ended"
        except Exception as error:
            # Type only: SDK errors can carry the join URL, which holds the access token.
            reason = f"stream failed ({type(error).__name__})"

        failures = 1 if delivered else failures + 1
        delay = backoff_delay(
            failures,
            base_seconds=policy.backoff_base_seconds,
            cap_seconds=policy.backoff_cap_seconds,
        )
        logger.warning(
            "live stream lost: %s after %d frames; rejoining in %.0f s", reason, delivered, delay
        )
        await sleep(delay)


class LiveFrameSource:
    """A `FrameSource` over a stream that is kept alive in the background."""

    def __init__(
        self, open_stream: OpenStream, policy: RejoinPolicy, *, clock: Clock = time.monotonic
    ) -> None:
        self._open_stream = open_stream
        self._policy = policy
        self._slot = LatestFrameSlot(clock)
        self._index = 0
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None

    def frames(self) -> Iterator[Frame]:
        """Yield frames as they arrive. Blocks, without ending, while the stream is down."""
        try:
            while True:
                frame = self.poll(POLL_SECONDS)
                if frame is not None:
                    yield frame
        finally:
            self.close()

    def poll(self, timeout_seconds: float) -> Frame | None:
        """The newest unread frame, or None if none arrives within the timeout."""
        self._ensure_started()
        image = self._slot.get(timeout_seconds)
        if image is None:
            return None
        frame = Frame(image=image, ts=datetime.now(UTC), index=self._index)
        self._index += 1
        return frame

    def seconds_since_last_frame(self) -> float:
        """How long the source has been blind. This is the health signal."""
        return self._slot.seconds_since_last_put()

    @property
    def dropped_frames(self) -> int:
        return self._slot.dropped

    def close(self) -> None:
        """Leave the stream and stop the background thread. Safe to call twice."""
        thread, loop, task = self._thread, self._loop, self._task
        if thread is None or loop is None or task is None:
            return
        self._thread = None
        loop.call_soon_threadsafe(task.cancel)
        thread.join(SHUTDOWN_TIMEOUT_SECONDS)
        if thread.is_alive():
            logger.warning(
                "live source thread did not stop within %.0f s", SHUTDOWN_TIMEOUT_SECONDS
            )

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        started = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(started,), name="poolguard-live-source", daemon=True
        )
        self._thread.start()
        started.wait()

    def _run(self, started: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        task = loop.create_task(pump(self._open_stream, self._slot, self._policy))
        self._loop, self._task = loop, task
        started.set()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            logger.info("live source stopped")
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
