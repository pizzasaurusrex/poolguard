"""The resilient live-source machinery, driven by scripted fake streams."""

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import numpy as np
import pytest

from poolguard.vision.frames import Frame
from poolguard.vision.live import (
    LatestFrameSlot,
    LiveFrameSource,
    RejoinPolicy,
    backoff_delay,
    pump,
)

STALE = 0.05
POLICY = RejoinPolicy(stale_after_seconds=STALE, backoff_base_seconds=1.0, backoff_cap_seconds=4.0)
HANG = "hang"
FAIL_OPEN = "fail-open"

Step = np.ndarray | str
OpenStream = Callable[[], AbstractAsyncContextManager[AsyncIterator[np.ndarray]]]


def image(value: int = 0) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


class ScriptedStreams:
    """Each open() plays the next script; the last script repeats forever."""

    def __init__(self, *scripts: list[Step]) -> None:
        self._scripts = scripts
        self.opened = 0
        self.closed = 0

    def open(self) -> AbstractAsyncContextManager[AsyncIterator[np.ndarray]]:
        script = self._scripts[min(self.opened, len(self._scripts) - 1)]
        self.opened += 1
        return self._session(script)

    @asynccontextmanager
    async def _session(self, script: list[Step]) -> AsyncIterator[AsyncIterator[np.ndarray]]:
        if script and isinstance(script[0], str) and script[0] == FAIL_OPEN:
            raise RuntimeError("wss://example/rtc?access_token=secret-token")
        try:
            yield self._play(script)
        finally:
            self.closed += 1

    async def _play(self, script: list[Step]) -> AsyncIterator[np.ndarray]:
        for step in script:
            if isinstance(step, str):
                await asyncio.sleep(3600)
            else:
                yield step


async def run_pump_until(
    streams: ScriptedStreams,
    slot: LatestFrameSlot,
    done: Callable[[], bool],
    delays: list[float] | None = None,
) -> None:
    async def record_sleep(seconds: float) -> None:
        if delays is not None:
            delays.append(seconds)
        await asyncio.sleep(0)

    task = asyncio.create_task(pump(streams.open, slot, POLICY, sleep=record_sleep))
    try:
        async with asyncio.timeout(2.0):
            while not done():
                await asyncio.sleep(0.005)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestBackoffDelay:
    def test_starts_at_the_base_and_doubles(self) -> None:
        delays = [backoff_delay(n, base_seconds=2.0, cap_seconds=60.0) for n in (1, 2, 3)]
        assert delays == [2.0, 4.0, 8.0]

    def test_never_exceeds_the_cap(self) -> None:
        assert backoff_delay(20, base_seconds=2.0, cap_seconds=60.0) == 60.0


class TestLatestFrameSlot:
    def test_get_returns_none_when_nothing_arrives_in_time(self) -> None:
        assert LatestFrameSlot().get(timeout_seconds=0.01) is None

    def test_a_newer_frame_replaces_an_unread_one(self) -> None:
        slot = LatestFrameSlot()
        slot.put(image(1))
        slot.put(image(2))

        latest = slot.get(timeout_seconds=0.01)
        assert latest is not None
        assert latest[0, 0, 0] == 2
        assert slot.dropped == 1

    def test_a_frame_is_handed_out_once(self) -> None:
        slot = LatestFrameSlot()
        slot.put(image(1))
        slot.get(timeout_seconds=0.01)
        assert slot.get(timeout_seconds=0.01) is None

    def test_get_wakes_when_a_frame_arrives_from_another_thread(self) -> None:
        slot = LatestFrameSlot()
        threading.Timer(0.02, slot.put, args=(image(7),)).start()
        latest = slot.get(timeout_seconds=1.0)
        assert latest is not None
        assert latest[0, 0, 0] == 7


class TestPump:
    def test_delivers_frames_to_the_slot(self) -> None:
        streams = ScriptedStreams([image(5), HANG])
        slot = LatestFrameSlot()
        seen: list[np.ndarray] = []

        def got_frame() -> bool:
            latest = slot.get(timeout_seconds=0)
            if latest is not None:
                seen.append(latest)
            return bool(seen)

        asyncio.run(run_pump_until(streams, slot, got_frame))
        assert seen[0][0, 0, 0] == 5

    def test_rejoins_when_a_connected_stream_goes_quiet(self) -> None:
        # The 2026-10-12 failure: the room stayed connected for 21 minutes with no frames.
        streams = ScriptedStreams([image(1), HANG], [image(2), HANG])
        asyncio.run(run_pump_until(streams, LatestFrameSlot(), lambda: streams.opened >= 2))
        assert streams.closed >= 1

    def test_rejoins_when_no_first_frame_ever_arrives(self) -> None:
        streams = ScriptedStreams([HANG])
        asyncio.run(run_pump_until(streams, LatestFrameSlot(), lambda: streams.opened >= 2))

    def test_rejoins_when_the_stream_ends(self) -> None:
        streams = ScriptedStreams([image(1)])
        asyncio.run(run_pump_until(streams, LatestFrameSlot(), lambda: streams.opened >= 3))

    def test_rejoins_when_opening_fails(self) -> None:
        streams = ScriptedStreams([FAIL_OPEN], [image(1), HANG])
        asyncio.run(run_pump_until(streams, LatestFrameSlot(), lambda: streams.opened >= 2))

    def test_backs_off_between_failures_and_resets_after_a_frame(self) -> None:
        streams = ScriptedStreams([FAIL_OPEN], [FAIL_OPEN], [FAIL_OPEN], [image(1)], [FAIL_OPEN])
        delays: list[float] = []
        asyncio.run(run_pump_until(streams, LatestFrameSlot(), lambda: len(delays) >= 5, delays))
        assert delays[:5] == [1.0, 2.0, 4.0, 1.0, 2.0]

    def test_a_failure_is_logged_without_its_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # SDK errors can carry the join URL, which holds the access token.
        streams = ScriptedStreams([FAIL_OPEN], [image(1), HANG])
        with caplog.at_level("WARNING"):
            asyncio.run(run_pump_until(streams, LatestFrameSlot(), lambda: streams.opened >= 2))
        assert "RuntimeError" in caplog.text
        assert "secret-token" not in caplog.text


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class TestLiveFrameSource:
    def test_yields_frames_with_an_aware_timestamp_and_running_index(self) -> None:
        streams = ScriptedStreams([image(3), HANG])
        source = LiveFrameSource(streams.open, POLICY)
        try:
            frame = next(source.frames())
        finally:
            source.close()

        assert isinstance(frame, Frame)
        assert frame.index == 0
        assert frame.ts.tzinfo is not None
        assert frame.image[0, 0, 0] == 3
        assert not frame.image.flags.writeable

    def test_poll_returns_none_while_the_stream_is_blind(self) -> None:
        source = LiveFrameSource(ScriptedStreams([HANG]).open, POLICY)
        try:
            assert source.poll(timeout_seconds=0.02) is None
        finally:
            source.close()

    def test_frame_age_counts_from_start_until_the_first_frame(self) -> None:
        clock = FakeClock()
        source = LiveFrameSource(ScriptedStreams([HANG]).open, POLICY, clock=clock)
        try:
            source.poll(timeout_seconds=0.01)
            clock.now += 200.0
            assert source.seconds_since_last_frame() == pytest.approx(200.0)
        finally:
            source.close()

    def test_frame_age_resets_when_a_frame_arrives(self) -> None:
        clock = FakeClock()
        source = LiveFrameSource(ScriptedStreams([image(1), HANG]).open, POLICY, clock=clock)
        try:
            clock.now += 50.0
            assert source.poll(timeout_seconds=1.0) is not None
            clock.now += 7.0
            assert source.seconds_since_last_frame() == pytest.approx(7.0)
        finally:
            source.close()

    def test_close_stops_the_background_thread_and_leaves_the_stream(self) -> None:
        streams = ScriptedStreams([image(1), HANG])
        source = LiveFrameSource(streams.open, POLICY)
        assert source.poll(timeout_seconds=1.0) is not None
        before = threading.active_count()
        source.close()
        assert threading.active_count() == before - 1
        assert streams.closed == 1
