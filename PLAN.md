# PLAN — progress journal

Update this file with every commit so a fresh session can resume. Phases are
defined in [PRD.md §10](PRD.md#10-phases). Sequencing changed 2026-07-29
(ADR-010): software is built and tested on a dev machine first; hardware is
ordered only when the P0 bench question (Hailo FPS) is the last one open.

## Current state (2026-07-29)

- PRD drafted (PRD.md), BOM with links (BOM.md), POC BOM (POC-BOM.md).
- Architecture doc with ADRs (docs/architecture.md) — 001–007, 010 accepted;
  008/009 deferred with revisit triggers.
- Repo public on GitHub (building in public): MIT license, safety disclaimer,
  secret scanning + push protection, Dependabot, main-branch ruleset.
- Python 3.12 + uv, src layout, frozen Pydantic event/config models, FPS
  benchmark script, tests for events and config.
- Vision seam (`PoseEstimator`/`FrameSource` protocols, Ultralytics dev
  backend) and replay CLI (`poolguard-replay`) landed — PR 2, PR 3.
- Tracking stage landed (PR 5): custom two-pass IoU tracker (`tracking.py`),
  `--track` flag; see [CLAUDE.md Decisions](CLAUDE.md#decisions) 2026-07-30.
- Annotated replay rendering (PR 6, open): `--render` flag draws per-track
  colored boxes and a "ghost box" for coasting tracks — the primary
  debugging tool for tracker behavior on labeled clips.
- **No hardware purchased. Hardware order is deliberately deferred (ADR-010)**
  until PD exit criteria are met.

## PD — dev-machine sandbox (no hardware, $0)  ⬅ current

Work happens in reviewable PRs against `main`.

- [ ] Vision seam: `PoseEstimator` protocol + Ultralytics dev backend,
      `FrameSource` protocol + video-file source (PR 2)
- [ ] Replay CLI: video file → pose → detection stream (PR 3)
- [ ] Test footage: collect public/self-shot pool clips; staged dry-land
      "distress" clip for rule smoke tests
- [x] Tracking (ByteTrack-style two-pass IoU, custom) over pose detections —
      `tracking.py`, `--track` on poolguard-replay (PR 5)
- [ ] Rules engine v0: distress / submersion / entry over track histories
      (thresholds in config)
- [ ] Replay-harness regression suite over labeled clips
- [ ] Local RTSP loop via mediamtx to exercise ingest/reconnect path
- [ ] PagerDuty Free signup + escalation drill — resolves ADR-006's named
      condition (voice calls on free tier?) with zero hardware
- [ ] Exit criteria: rules engine produces correct events on labeled test
      clips end-to-end on the dev machine; alert manager works with mock siren

## P0 — bench rig (hardware validation only)

Narrowed by ADR-010: the only open questions are Hailo throughput and the
Pi-specific paths. Order hardware (POC-BOM.md) only after PD exits.

- [ ] Order POC hardware (~$317, POC-BOM.md)
- [ ] Flash Pi OS Lite, install Hailo runtime (`hailo-all`)
- [ ] Hailo backend for the `PoseEstimator` seam (`.hef` via HailoRT)
- [ ] Run the PD replay suite on-target; re-tune thresholds for INT8 deltas
- [ ] Exit criterion: 15+ FPS pose inference at 1080p (looped RTSP source
      per ADR-002 addendum; no camera purchase yet)

## P1 — pool install + footage library

- [ ] Buy camera + PoE injector (deferred BOM items) after P0 passes
- [ ] Site survey: mount point, cable run length (finalize BOM items 11–12)
- [ ] Mount camera, run cable, weatherize
- [ ] Recording harness → 2 weeks of sun/rain/night/swimmer footage
- [ ] Label clip library; fold real footage into the PD replay suite

## P2 — detection + rules tuning on real footage

- [ ] Re-tune tracking + rules thresholds against P1 footage
- [ ] Staged scenario tests (mock distress, weighted mannequin, doll entry)

## P3 — alerting

- [ ] Siren via GPIO relay; armed/swim/maintenance modes
- [ ] ntfy push with snapshot; Home Assistant via MQTT
- [ ] PagerDuty emergency escalation (per ADR-006; Twilio fallback)
- [ ] End-to-end staged drill: < 5 s to siren

## P4 — hardening

- [ ] Watchdog, self-test heartbeat, degraded-mode alerts
- [ ] One month live tuning; false-alarm budget ≤ 1/week

## P5 (v2)

- [ ] Underwater camera trade study: over-the-edge housing vs floating buoy
- [ ] Open-source release: docs, install guide, liability review
