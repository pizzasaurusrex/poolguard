# PoolGuard

Open-source AI drowning detection for residential pools. The design: a PoE
camera and a Raspberry Pi 5 with a Hailo NPU watch the pool 24/7 for surface
distress, submersion, and unsupervised entry, and raise tiered alerts, from a
phone push up to an on-site alarm and call escalation. Most of that is not
built yet; see [Status](#status).

> **PoolGuard is a supplement to supervision, fencing, and door alarms — never
> a substitute.** It is not a certified life-safety device. No detection
> system removes the need for an attentive adult.

## ⚠️ Safety disclaimer

This project is **in development and has never been validated against real
drowning scenarios**. Do not install it and trust it — at this stage it is a
design and a scaffold, not a working detector.

- PoolGuard is **not a certified life-safety device** (no UL 2621 or
  equivalent certification) and is provided **without warranty of any kind**
  (see [LICENSE](LICENSE)).
- It is one layer in the
  [CPSC "layers of protection"](https://www.cpsc.gov/Safety-Education/Safety-Education-Centers/Pool-Safely)
  model. Fencing, self-latching gates, door alarms, and — above all —
  **attentive adult supervision** come first. No camera system replaces any
  of them.
- Detection can and will fail: glare, rain, night conditions, occlusion,
  hardware faults, and software bugs are all expected failure modes. Design
  details and known blind spots are documented in
  [docs/architecture.md](docs/architecture.md) and the
  [decision log](docs/decision-log.md).

If you build this, you accept full responsibility for its use.

## Status

**Building in public** — this repo is live from day one, before the hardware
has even arrived. Expect broken states, unproven rules, and revised decisions.

Currently: **PD phase — no hardware, everything runs on a dev machine**
(ADR-010).

| Piece | State |
|---|---|
| Pose detection on video files | Works, via the replay CLI below |
| Tracking (persistent IDs, coasting through dropouts) | Works |
| Annotated replay video with HUD | Works |
| Regression suite over labeled footage | Tracker assertions run; event assertions wait on the rules engine |
| ntfy push and heartbeat alert channels | Built and testable with `poolguard-alert-test`; not yet driven by a live pipeline |
| Rules engine (distress / submersion / entry) | **Not built — next** |
| Siren, Home Assistant, call escalation | Not built |
| Live camera pipeline, Hailo backend | Not built; no hardware purchased |

So today the code can show you who it sees in a clip and for how long it has
lost them. It cannot yet decide that anyone is in trouble.

Where to read more:

- [PRD.md](PRD.md): requirements and open questions
- [docs/architecture.md](docs/architecture.md): architecture and the ADRs
  behind each major choice
- [docs/decision-log.md](docs/decision-log.md): dated decisions, findings and
  measurements, including what was tried and rejected
- [PLAN.md](PLAN.md): current progress
- [BOM.md](BOM.md): parts list
- [docs/edge-inference.md](docs/edge-inference.md): how the AI HAT+ / model
  pipeline works

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). No Raspberry Pi
or camera is needed — the whole pipeline runs on a laptop (ADR-010).

```sh
uv sync              # core deps + dev tools
uv run pytest        # unit tests
uv run ruff check    # lint
```

The regression suite replays labeled pool footage through a real pose model,
so it is slow and opt-in:

```sh
uv sync --extra vision
uv run pytest -m regression
```

Ground truth is in [tests/labels/shots.yaml](tests/labels/shots.yaml). The
footage itself is not in the repo, and a shot whose file is missing from
`clips/` is skipped, so a fresh checkout passes with every shot skipped.

## Try it: pose detection on a video clip

Install the vision extra (opencv + ultralytics; on Apple Silicon inference
uses MPS automatically), then replay any video file through the pose
pipeline:

```sh
uv sync --extra vision
uv run poolguard-replay --video path/to/clip.mp4
```

Output is a summary line (frames, frames with people, detection count,
throughput); add `--per-frame` to see each frame's detections, or
`--min-confidence 0.4` to raise the detection floor (it defaults to the
tracker's `low_confidence`, 0.1, so weak detections reach the rescue pass). The first run
downloads the pose model (~6 MB) to the working directory; weights are
never committed (see License).

A phone video of people walking nearby is enough to see detections. Pool
clips are what the PD phase collects for rule development.

**If you get no detections on wide or distant footage,** that is a known
limit rather than a broken install. The CLI runs the model at the frame's
own resolution, capped at 1920 px (`--max-inference-size`). Lowering that cap
is faster but costs detections: on 1080p footage from a real backyard mount
(people 35–100 px tall) a 640 px input detected nobody at all, 1280 reached
66% of frames and 1920 reached 87%. People who are
fully underwater are also close to invisible to this model. Both findings are
in the [decision log](docs/decision-log.md).

## Try it: tracking and annotated replay

Add `--track` to run detections through the tracking stage (persistent IDs,
low-confidence "rescue" matching, coasting through brief detector dropouts)
and the rules engine. Each event the rules raise is printed as an `EVENT`
line with its time into the clip, and the summary ends with an event count.
Today the only rule is submersion: a person last seen in the water who stays
unseen for 12 s.

```sh
uv run poolguard-replay --video path/to/clip.mp4 --track --per-frame
```

Add `--render out.mp4` (implies `--track`) to write an annotated video
instead — colored box per track ID, a distinct "ghost box" with an elapsed
timer for tracks the detector has temporarily lost, and the configured pool
zone outline. A HUD shows the clip clock, live and coasting track counts, the
running detection rate, and a meter filling toward the submersion timeout:

```sh
uv run poolguard-replay --video path/to/clip.mp4 --render out.mp4
```

Add `--labels tests/labels/shots.yaml` to show the expected event for that
clip on the HUD, when the manifest has an entry matching the file name.

This is the primary debugging tool for tracker behavior — watch the ghost
box hold a person's identity through a splash or brief occlusion instead of
inferring it from log lines.

## Try it: alert channels

Two alert channels exist so far: an [ntfy](https://ntfy.sh) push notification
and a heartbeat ping to an [Uptime Kuma](https://github.com/louislam/uptime-kuma)
push monitor, which reports when the pipeline stops pinging. Set either or
both in the environment or in `.env`:

```sh
POOLGUARD_ALERT_NTFY_TOPIC_URL=https://ntfy.example.com/your-topic
POOLGUARD_ALERT_HEARTBEAT_PUSH_URL=https://kuma.example.com/api/push/your-token
```

Both URLs are credentials (the ntfy topic name is its only secret), so keep
them out of version control. Then send one real test message through each
configured channel:

```sh
uv run poolguard-alert-test
```

An unset channel is reported as skipped, and the command exits non-zero if a
configured channel fails. No camera or vision extra is needed.

## Other utilities

```sh
# P0 bench: measure raw RTSP capture FPS (needs a live stream).
# The URL holds the camera password, so it comes from the environment or .env,
# never a command-line flag.
POOLGUARD_CAMERA_RTSP_URL=rtsp://user:pass@cam/h264Preview_01_main uv run poolguard-benchmark
```

## Check that the camera has not moved

The pool zone is traced for one camera angle. If the camera is re-mounted the
outline no longer sits on the water, and nothing else would notice. Record the
view's fixed landmarks once, right after tracing the zone, then check any later
recording against them:

```sh
uv sync --extra vision --extra labels
uv run poolguard-view-check save --video clips/today.mp4 \
    --reference calibration/backyard.viewref.npz \
    --labels tests/labels/shots.yaml --camera backyard_simplisafe
uv run poolguard-view-check check --video clips/later.mp4 \
    --reference calibration/backyard.viewref.npz \
    --labels tests/labels/shots.yaml --camera backyard_simplisafe
```

`check` prints `aligned`, `moved` or `unknown`. Anything but `aligned` prints a
`system_degraded` event and exits non-zero. Without `--labels` and `--camera`
the zone comes from `POOLGUARD_TRACKING_POOL_ZONE`.

The reference file holds landmark positions and descriptors, not a picture,
but it still describes a private scene: `calibration/` is gitignored and the
file should never be committed.

## Layout

```
src/poolguard/
  events.py      # immutable Pydantic models passed between pipeline stages
  config.py      # validated runtime settings (env / .env)
  tracking.py    # greedy box-overlap tracker: Detections -> persistent TrackedPerson tracks
  camera_view.py # verdict on whether the camera still sees the traced view, and its alert
  rules.py       # rules engine: TrackedPerson tracks -> SafetyEvents (submersion so far)
  replay.py      # offline pipeline loop over a FrameSource (PD test harness)
  labels.py      # schema for the shot-label manifest (dev only, needs PyYAML)
  vision/        # frame + pose seams: protocols, video-file source, Ultralytics backend,
                 # annotated-replay renderer and HUD
  alerts/        # one module per alert channel: ntfy push, Uptime Kuma heartbeat
  scripts/       # CLI entry points (replay, alert test, FPS benchmark)
  main.py        # live pipeline entry point (later phase)
tests/
  labels/shots.yaml   # ground truth for the regression suite
```

## License

[MIT](LICENSE). Note: pose model weights are **not** distributed with this
repo — they are fetched from the Hailo Model Zoo at install time and carry
their own license (see [docs/edge-inference.md](docs/edge-inference.md)).
