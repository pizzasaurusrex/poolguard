"""SimpliSafe Outdoor Cam as a live frame source. Bench-only (ADR-002).

Requires the `simplisafe` extra (livekit, simplisafe-python). This rides
SimpliSafe's private API and their cloud, so it can break without notice and
it stops when the internet does. It exists to test the pipeline on live
backyard video, never to be the production camera.

How a stream is opened, as measured on 2026-10-12:
1. Log in with the stored refresh token and save the rotated one.
2. Ask the `live-view` endpoint for a LiveKit URL and room token. This call
   is also what wakes the battery camera.
3. Join the room. It is empty at first; the camera joins some seconds later.

Resilience is not handled here. `LiveFrameSource` reopens the stream whenever
frames stop, which repeats all three steps.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

import aiohttp
import numpy as np
from livekit import rtc
from pydantic import SecretStr
from simplipy import API
from simplipy.system import System

from poolguard.config import SimpliSafeSettings
from poolguard.vision.live import LiveFrameSource, RejoinPolicy
from poolguard.vision.simplisafe_auth import (
    LiveKitCredentials,
    RefreshTokenStore,
    parse_live_view,
)

logger = logging.getLogger(__name__)

LIVE_VIEW_URL = "https://app-hub.prd.aser.simplisafe.com/v2/cameras/{uuid}/{system_id}/live-view"
HTTP_OK = 200


class SimpliSafeStreamError(RuntimeError):
    """The stream could not be opened. Messages never include a token or URL."""


def simplisafe_frame_source(settings: SimpliSafeSettings) -> LiveFrameSource:
    """Build the source, failing now, not in the retry loop, if it is misconfigured."""
    if settings.camera_name is None:
        raise SimpliSafeStreamError("set POOLGUARD_SIMPLISAFE_CAMERA_NAME to the camera's name")
    store = RefreshTokenStore(settings.token_path)
    store.load()
    policy = RejoinPolicy(
        stale_after_seconds=settings.stale_rejoin_seconds,
        backoff_base_seconds=settings.rejoin_backoff_base_seconds,
        backoff_cap_seconds=settings.rejoin_backoff_cap_seconds,
    )
    return LiveFrameSource(partial(open_simplisafe_stream, settings.camera_name, store), policy)


@asynccontextmanager
async def open_simplisafe_stream(
    camera_name: str, store: RefreshTokenStore
) -> AsyncIterator[AsyncIterator[np.ndarray]]:
    credentials = await _fetch_credentials(camera_name, store)
    room = rtc.Room()
    video_track: asyncio.Future[rtc.Track] = asyncio.get_running_loop().create_future()

    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        _subscribe_if_video(publication)

    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO and not video_track.done():
            video_track.set_result(track)

    room.on("track_published", on_track_published)
    room.on("track_subscribed", on_track_subscribed)
    # Video only: the camera also publishes its microphone, which nothing here needs.
    await room.connect(
        credentials.url,
        credentials.token.get_secret_value(),
        rtc.RoomOptions(auto_subscribe=False),
    )
    try:
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                _subscribe_if_video(publication)
        logger.info("joined the SimpliSafe room; waiting for the camera's video")
        yield _images(video_track)
    finally:
        await room.disconnect()


def _subscribe_if_video(publication: rtc.RemoteTrackPublication) -> None:
    if publication.kind == rtc.TrackKind.KIND_VIDEO:
        publication.set_subscribed(True)


async def _images(video_track: asyncio.Future[rtc.Track]) -> AsyncIterator[np.ndarray]:
    stream = rtc.VideoStream(await video_track)
    try:
        async for event in stream:
            yield to_bgr(event.frame)
    finally:
        await stream.aclose()


def to_bgr(frame: rtc.VideoFrame) -> np.ndarray:
    rgb = frame.convert(rtc.VideoBufferType.RGB24)
    pixels = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, 3)
    return np.ascontiguousarray(pixels[:, :, ::-1])


async def _fetch_credentials(camera_name: str, store: RefreshTokenStore) -> LiveKitCredentials:
    async with aiohttp.ClientSession() as session:
        api = await API.async_from_refresh_token(store.load().get_secret_value(), session=session)
        if api.refresh_token is None or api.access_token is None:
            raise SimpliSafeStreamError("SimpliSafe login returned no tokens")
        # Saved before anything else can fail: the old token is already spent.
        store.save(SecretStr(api.refresh_token))
        url = _live_view_url(await api.async_get_systems(), camera_name)
        headers = {"Authorization": f"Bearer {api.access_token}"}
        async with session.get(url, headers=headers) as response:
            if response.status != HTTP_OK:
                raise SimpliSafeStreamError(f"live-view returned HTTP {response.status}")
            return parse_live_view(await response.json())


def _live_view_url(systems: dict[int, System], camera_name: str) -> str:
    for system_id, system in systems.items():
        for uuid, camera in system.cameras.items():
            if camera.name == camera_name:
                return LIVE_VIEW_URL.format(uuid=uuid, system_id=system_id)
    raise SimpliSafeStreamError(f"no SimpliSafe camera named {camera_name!r} on this account")
