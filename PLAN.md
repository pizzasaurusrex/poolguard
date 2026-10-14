# PLAN — progress journal

Update this file with every commit so a fresh session can resume. Phases are
defined in [PRD.md §10](PRD.md#10-phases). Sequencing changed 2026-07-29
(ADR-010): software is built and tested on a dev machine first; hardware is
ordered only when the P0 bench question (Hailo FPS) is the last one open.

## Current state (2026-10-11)

- PRD drafted (PRD.md), BOM with links (BOM.md), POC BOM (POC-BOM.md).
- Architecture doc with ADRs (docs/architecture.md) — 001–007, 010 accepted;
  008/009 deferred with revisit triggers.
- Repo public on GitHub (building in public): MIT license, safety disclaimer,
  secret scanning + push protection, Dependabot, main-branch ruleset.
- Python 3.12 + uv, src layout, frozen Pydantic event/config models, FPS
  benchmark script, tests for events and config.
- Vision seam (`PoseEstimator`/`FrameSource` protocols, Ultralytics dev
  backend) and replay CLI (`poolguard-replay`) landed — PR 2, PR 4.
- Tracking stage landed (PR 5): custom two-pass IoU tracker (`tracking.py`),
  `--track` flag; see the [decision log](docs/decision-log.md) 2026-07-30.
- Annotated replay rendering (PR 6): `--render` flag draws per-track
  colored boxes and a "ghost box" for coasting tracks — the primary
  debugging tool for tracker behavior on labeled clips.
- Labeled regression suite, replay HUD and camera profiles landed (PR 8):
  9 shots labeled in `tests/labels/shots.yaml`, tracker assertions active.
- First alert channels landed (PR 9): ntfy push and Uptime Kuma heartbeat
  under `alerts/`, plus `poolguard-alert-test`. Nothing drives them from a
  live pipeline yet.
- Decision history moved out of CLAUDE.md into
  [docs/decision-log.md](docs/decision-log.md) (PR 10).
- First real footage from the backyard camera (2026-08-25) showed input
  resolution is the dominant detection variable: 0% of frames at the default
  640 px input, 87% at 1920. Not yet wired into the pipeline.
- **No hardware purchased. Hardware order is deliberately deferred (ADR-010)**
  until PD exit criteria are met.

## PD — dev-machine sandbox (no hardware, $0)  ⬅ current

Work happens in reviewable PRs against `main`.

- [x] Vision seam: `PoseEstimator` protocol + Ultralytics dev backend,
      `FrameSource` protocol + video-file source (PR 2)
- [x] Replay CLI: video file → pose → detection stream (PR 4)
- [ ] Test footage: collect public/self-shot pool clips; staged dry-land
      "distress" clip for rule smoke tests
- [x] Tracking (ByteTrack-style two-pass IoU, custom) over pose detections —
      `tracking.py`, `--track` on poolguard-replay (PR 5)
- [~] Rules engine v0: distress / submersion / entry over track histories
      (thresholds in config). Submersion landed (`rules.py`, runs under
      `--track`); distress and entry are not written
      - **Resolved (provisional): submersion timeout 15 s → 12 s** so the POC
        clip library can exercise the rule at all. Fitted to footage, not
        derived from safety — must be re-derived on real P1 footage against
        the false-alarm budget (kids dive under routinely).
      - **Open: clip library is too short.** Every clip shows its event from
        frame one (no calm lead-in), so the earliest possible fire is exactly
        the rule threshold and a clip can only test its rule if
        duration > threshold. At current defaults only 2 of 6 positives
        qualify: `activedrowning_01` (7.5 s) and `distress_05` (14.6 s), both
        distress. **No clip can currently test submersion at all** — the
        closest, `fullySubmerged_08` at 13.8 s, misses the 15 s threshold by
        1.2 s. Sourcing longer cuts (especially of `childFallsIn_03`) is a
        prerequisite for trusting the submersion rule.
- [~] Replay-harness regression suite over labeled clips — schema, manifest and
      harness landed (`labels.py`, `tests/labels/shots.yaml`,
      `tests/test_clip_regression.py`, `-m regression`). Tracker assertions
      (miss + fragmentation) and event assertions (verdict + onset window)
      are both active. No labeled shot is a positive yet: the backyard clips
      are all too short for the 12 s submersion rule to conclude.
- [x] Annotated replay HUD (`vision/hud.py`): per-frame clock, live/coasting
      counts, running detection rate, a submersion meter filling toward the
      timeout, the label's expected event, and a priority status line.
      `poolguard-replay --render OUT.mp4 --labels tests/labels/shots.yaml`
- [ ] **Detection is the bottleneck, not tracking.** Underwater/submerged
      swimmers are near-invisible to the land-trained pose model (2% det on
      `fullySubmerged_08`), and in `fallsInDrowns_09` the child *out* of the
      water tracks cleanly while the drowning child drops in and out. Evaluate
      before more rules work: a swimmer-specific detector, plain person
      detection (box only, no pose) as a fallback when keypoints fail, or
      background subtraction / motion as a water-disturbance signal.
      - **Found 2026-08-25: input resolution matters more than the water.**
        On 1080p backyard footage, detection goes 0% → 66% → 87% of frames at
        `imgsz` 640 / 1280 / 1920. Tiled inference was benchmarked and
        rejected for now (gains were mostly shadow false positives). See the
        [decision log](docs/decision-log.md).
- [~] Inference near native resolution: the Ultralytics backend now runs at
      the frame's own size, capped at 1920 (`--max-inference-size` on the
      replay CLI). Still open: crop to the camera profile's `crop_rect`
- [x] Per-clip `pool_zone` — `TrackingSettings.pool_zone` is a polygon and
      `in_water` is a point-in-polygon test; `poolguard-replay --labels`
      applies the shot's camera profile. Not yet checked against real
      backyard footage
- [ ] Pool zone survives a camera re-mount: the outline is traced for one
      angle and the battery camera moves every time it is charged
      ([#23](https://github.com/pizzasaurusrex/poolguard/issues/23))
- [ ] Jump-in false alarm: the track left at the pool edge when a swimmer
      jumps in times out as a submersion (seen on `drowning-test.mp4`)
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
- [~] ntfy push with snapshot; Home Assistant via MQTT — ntfy channel built
      (PR 9, text only, no snapshot), not yet driven by an alert manager
- [ ] PagerDuty emergency escalation (per ADR-006; Twilio fallback)
- [ ] End-to-end staged drill: < 5 s to siren

## P4 — hardening

- [~] Watchdog, self-test heartbeat, degraded-mode alerts — heartbeat
      pinger built (PR 9), not yet called by a running pipeline
- [ ] One month live tuning; false-alarm budget ≤ 1/week

## P5 (v2)

- [ ] Underwater camera trade study: over-the-edge housing vs floating buoy
- [ ] Open-source release: docs, install guide, liability review
