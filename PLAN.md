# PLAN — progress journal

Update this file with every commit so a fresh session can resume. Phases are
defined in PRD.md §10.

## Current state (2026-07-24)

- PRD drafted (PRD.md), BOM with links (BOM.md), repo scaffolded.
- Architecture doc with ADRs (docs/architecture.md) — decisions 001–007
  recorded, 008/009 deferred with revisit triggers.
- Python 3.12 + uv, src layout, Pydantic event/config models, FPS benchmark
  script, tests for event models.
- **No hardware purchased yet.** Next action is ordering Phase 0 items
  (BOM.md items 1–6).

## P0 — bench rig  ⬅ current

- [ ] Order Phase 0 hardware (BOM items 1–6)
- [x] Repo scaffold (events, config, benchmark script, tests)
- [ ] Flash Pi OS Lite, install Hailo runtime (`hailo-all`)
- [ ] Camera on bench: set stream to H.265 (Pi 5 has no HW H.264 decode),
      RTSP reachable, run `poolguard-benchmark` → capture FPS
- [ ] Run pretrained YOLO pose via Hailo examples on the live stream
- [ ] Exit criterion: 15+ FPS pose inference at 1080p

## P1 — pool install + footage library

- [ ] Site survey: mount point, cable run length (finalize BOM items 11–12)
- [ ] Mount camera, run cable, weatherize
- [ ] Recording harness → 2 weeks of sun/rain/night/swimmer footage
- [ ] Label clip library; build replay harness for offline pipeline testing

## P2 — detection + rules

- [ ] Tracking (ByteTrack) over pose detections
- [ ] Rules engine: distress / submersion / entry (thresholds in config)
- [ ] Replay-harness test suite over labeled clips
- [ ] Staged scenario tests (mock distress, weighted mannequin, doll entry)

## P3 — alerting

- [ ] Siren via GPIO relay; armed/swim/maintenance modes
- [ ] ntfy push with snapshot; Home Assistant via MQTT
- [ ] Twilio SMS→call escalation with ack flow
- [ ] End-to-end staged drill: < 5 s to siren

## P4 — hardening

- [ ] Watchdog, self-test heartbeat, degraded-mode alerts
- [ ] One month live tuning; false-alarm budget ≤ 1/week

## P5 (v2)

- [ ] Underwater camera trade study: over-the-edge housing vs floating buoy
- [ ] Open-source release: docs, install guide, liability review
