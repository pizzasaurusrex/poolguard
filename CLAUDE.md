# PoolGuard

AI-based residential pool drowning detection: PoE camera + Raspberry Pi 5 with
Hailo NPU running pose-based detection, tiered alerting (siren/push/HA/Twilio).
See PRD.md for full requirements.

## Decisions

- 2026-07-24: Hardware budget ≤ $500; target BOM ~$460 ([PRD §6](PRD.md#6-hardware)).
- 2026-07-24: Compute = Pi 5 + AI HAT+ (Hailo) over Jetson Orin Nano — cost and ecosystem; Jetson kept as escape hatch since camera/alert layers are compute-agnostic.
- 2026-07-24: v1 is above-water camera only; underwater camera (over-the-edge housing, no wall penetration) deferred to v2 for bottom-of-pool confirmation.
- 2026-07-24: All inference on-device, no cloud — privacy, latency, siren must work without internet.
- 2026-07-24: Sensitivity-biased tuning — accept ~1 false siren/week rather than risk missed detections; tiered alerts (watch/warn/emergency) manage alert fatigue.
- 2026-07-24: Interpretable rules engine on top of pose+tracking, not end-to-end black-box classification.
- 2026-07-24: Leaning custom Python pipeline (Pydantic-typed events) with Frigate only as companion NVR — still open, see [PRD §11](PRD.md#11-open-questions).
- 2026-07-24: Language = Python 3.12+, uv for env/deps, hatchling build, src layout, ruff for lint. Heavy vision deps (opencv, ultralytics) isolated in a `vision` extra so dev machines stay light.
- 2026-07-24: Pipeline stages communicate via frozen (immutable) Pydantic models in `events.py`; config validated at startup via pydantic-settings.
- 2026-07-24: BOM buys the 26 TOPS AI HAT+ over 13 TOPS — $40 for pose-model headroom and a v2 second camera.
- 2026-07-24: Repo is public from day one (building in public) at github.com/pizzasaurusrex/poolguard. MIT license; model weights are never committed — fetched from Hailo Model Zoo at install time, sidestepping the AGPL redistribution conflict (ADR-009 still governs the *recommended* model before P5). README carries a prominent safety disclaimer since the project is discoverable while unvalidated.
- 2026-07-24: pydantic-settings composition pattern: nested sections are BaseSettings wired with `Field(default_factory=...)` so each reads its own `POOLGUARD_<SECTION>_*` env vars at construction (plain nesting silently bypasses child env machinery — caught as a real bug). Credentials in config use `SecretStr` (never plain str); all timestamps are `AwareDatetime`; settings classes are frozen like the event models.
- 2026-07-28: Siren-as-primary-attention-grabber reopened after external feedback — loud-party swim mode can drown out audio; visual alert at the mount (strobe/LED/flag) under evaluation as the swim-mode channel ([PRD §11](PRD.md#11-open-questions)). v1 alert semantics stay "look at the pool" — no per-swimmer localization (backyard pools are covered by a single glance).
- 2026-07-28: "Drowning doesn't look like drowning" documented as the core detection principle; distress-vs-treading-water is named the central discrimination problem ([PRD §5](PRD.md#5-system-overview)).
- 2026-07-24: Emergency-tier escalation via PagerDuty Free (Events API, family as responders) instead of hand-rolled Twilio ladder — free ack/retry/escalation logic plus critical-alert DND bypass on phones. ntfy stays for watch/warn tiers; Twilio demoted to fallback. Pending verification that free tier includes voice calls ([PRD §11](PRD.md#11-open-questions)).
