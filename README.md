# PoolGuard

Open-source AI drowning detection for residential pools. A PoE camera and a
Raspberry Pi 5 with a Hailo NPU watch the pool 24/7 for surface distress,
submersion, and unsupervised entry, and fire tiered alerts: local siren →
phone push → smart home → SMS/call escalation.

> **PoolGuard is a supplement to supervision, fencing, and door alarms — never
> a substitute.** It is not a certified life-safety device. No detection
> system removes the need for an attentive adult.

## Status

Pre-hardware. PRD drafted; scaffold only. See [PRD.md](PRD.md) for the full
design, [BOM.md](BOM.md) for the parts list, and [PLAN.md](PLAN.md) for
current progress.

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

MIT
