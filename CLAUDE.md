# PoolGuard

AI-based residential pool drowning detection: PoE camera + Raspberry Pi 5 with
Hailo NPU running pose-based detection, tiered alerting (siren/push/HA/Twilio).

## Where things live

| Need | Read |
|---|---|
| Requirements, success criteria, open questions | [PRD.md](PRD.md) |
| Architecture and ADRs (why each major choice was made) | [docs/architecture.md](docs/architecture.md) |
| Dated decisions, findings and measurements | [docs/decision-log.md](docs/decision-log.md) |
| Phase progress and what is next | [PLAN.md](PLAN.md) |
| Footage requirements | [docs/footage-spec.md](docs/footage-spec.md) |
| Hailo / NPU details | [docs/edge-inference.md](docs/edge-inference.md) |
| Hardware | [BOM.md](BOM.md), [POC-BOM.md](POC-BOM.md) |

Read the decision log before changing the tracker, detection settings, label
schema, settings classes or alert channels: it records what was already tried
and rejected.

## Logging decisions

- Append new decisions, findings and measurements to `docs/decision-log.md` as a dated entry.
- Add a line to Standing rules below only when it changes how code gets written on a typical task. Keep it to one line; the reasoning stays in the log.
- A reversed architectural decision gets a superseding ADR in `docs/architecture.md`.

## Standing rules

### Workflow

- Work in feature branches and PRs against `main`; no direct commits to `main` for substantive changes.
- The repo is public. Never commit secrets, ntfy topic URLs, model weights, footage or camera view references. Weights are fetched at install time.
- Python 3.12+, uv for env and deps, hatchling build, src layout, ruff for lint.

### Design

- All inference runs on-device. The siren path must work with the internet down.
- Tuning is sensitivity-biased: a missed detection is worse than a false alarm (budget is about one false siren a week).
- Detection logic is an interpretable rules engine over pose and tracking, with thresholds in config. No end-to-end classifier.
- Inference sits behind the `PoseEstimator` protocol and frame input behind `FrameSource`. New backends and sources implement those seams.
- Bench-only video sources (Nest, SimpliSafe) never become the production camera.

### Code

- Pipeline stages communicate through frozen Pydantic models in `events.py`. Settings classes are frozen too, and `Frame` pixel data is read-only.
- All timestamps are `AwareDatetime`. Credentials are `SecretStr`, including the ntfy topic URL.
- Each settings section is its own `BaseSettings`, wired into the parent with `Field(default_factory=...)`, reading `POOLGUARD_<SECTION>_*` and using the shared `_SECTION_CONFIG`. Plain nesting silently skips env vars and `.env`.
- The core package never imports cv2, ultralytics or PyYAML. Code needing them lives in modules core does not import (`vision/video.py`, `vision/ultralytics_pose.py`, `labels.py`). A new optional dependency gets its own extra and its own isolated module.
- The tracker is a pure function, `advance(state, …) -> (new_state, people)`. Coasting tracks are first-class output: `seconds_since_last_seen` is the submersion signal.
- The rules engine is a pure function too, `evaluate(state, people, now, mode, settings) -> (new_state, events)`. Events are edge-triggered: raised once per incident, never once per frame.
- Alert channels are one module each under `alerts/`: a pure `build_message` plus a notifier taking an injected `httpx.Client`. Raise delivery errors `from None`, since httpx puts the request URL in its exception text.
- A monitoring failure (heartbeat ping) is logged and detection continues.

### Testing and tuning

- Regression tests are behind the `regression` pytest marker and deselected by default; run them with `-m regression`. A shot whose footage is absent locally skips.
- Labels are per shot in `tests/labels/shots.yaml`. `expected_event` and `onset_window` are set together or not at all. Camera geometry lives in a named `CameraProfile`, with the pool zone as a polygon.
- Loosening `MAX_FRAGMENTATION` is a regression that must be argued for in review.
- `submersion_timeout_seconds` (12.0) is provisional and was fitted to clip lengths. Re-derive it on real footage before it ships.
- Run inference near native resolution: `imgsz=640` detects nothing on the 1080p backyard footage.
- Submersion is only detectable if the person was tracked above water first, so test footage needs a calm lead-in.
- `detections/frame` is not a quality metric. Measure precision and recall against ground-truth boxes.
