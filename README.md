# PoolGuard

Open-source AI drowning detection for residential pools. A PoE camera and a
Raspberry Pi 5 with a Hailo NPU watch the pool 24/7 for surface distress,
submersion, and unsupervised entry, and fire tiered alerts: local siren →
phone push → smart home → SMS/call escalation.

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
  [docs/architecture.md](docs/architecture.md).

If you build this, you accept full responsibility for its use.

## Status

**Building in public** — this repo is live from day one, before the hardware
has even arrived. Expect broken states, unproven rules, and revised decisions
(recorded as ADRs in [docs/architecture.md](docs/architecture.md)).

Currently: pre-hardware. PRD drafted; scaffold only. See [PRD.md](PRD.md) for the full
design, [BOM.md](BOM.md) for the parts list, [PLAN.md](PLAN.md) for current
progress, and [docs/edge-inference.md](docs/edge-inference.md) for how the
AI HAT+ / model pipeline works.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync              # core deps + dev tools
uv run pytest        # tests
uv run ruff check    # lint

uv sync --extra vision   # adds opencv/ultralytics (needed on the Pi / for benchmarks)
uv run poolguard-benchmark --rtsp-url rtsp://user:pass@cam/h264Preview_01_main
```

## Layout

```
src/poolguard/
  events.py      # immutable Pydantic models passed between pipeline stages
  config.py      # validated runtime settings (env / .env)
  scripts/       # P0/P1 utilities (FPS benchmark, capture)
  main.py        # pipeline entry point (P2)
tests/
```

## License

[MIT](LICENSE). Note: pose model weights are **not** distributed with this
repo — they are fetched from the Hailo Model Zoo at install time and carry
their own license (see [docs/edge-inference.md](docs/edge-inference.md)).
