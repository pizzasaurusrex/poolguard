# POC BOM — P0 bench rig, cost-minimized

Companion to [BOM.md](BOM.md). This is the minimum spend to validate the P0
exit criterion (15+ FPS pose inference at 1080p) before committing to
Phase 3+ install hardware. Verified 2026-07-29.

## What's different from BOM.md's Phase 0 list

No camera purchase. No PoE injector. The P0 exit criterion only needs an
RTSP-capable 1080p source reaching the Pi over the LAN — it doesn't care
whether that source is the eventual production camera. Any of the options
below satisfy it at $0 marginal cost:

- **Old Nest Cam (Indoor/Outdoor/IQ)** — confirmed 1080p @ 30fps. RTSP is
  available via Google's Smart Device Management API (cloud-mediated, token
  expires every 5 min, needs periodic re-extension). Fine for a bench test;
  not appropriate for the deployed system (violates ADR-002's
  no-cloud-in-detection-path principle — noted so this doesn't get forgotten
  and carried into P1 by accident).
- **Phone as IP camera** — IP Webcam (Android) or equivalent iOS app,
  streams RTSP over local WiFi for free.
- **Looped test footage** — ffmpeg can serve a local video file as a fake
  RTSP source. Hailo's hailo-rpi5-examples repo ships sample clips; this is
  also the cleanest option since it's reproducible and doubles as an early
  input for the P2 replay harness.

## POC BOM — items to actually buy

| # | Item | Price | Source | Notes |
|---|---|---|---|---|
| 1 | Raspberry Pi 5, 8GB (bare board) | $175.00 | PiShop.us | Micro Center has it for $169.99, in-store pickup only (Santa Clara/Tustin both in stock) |
| 2 | Raspberry Pi AI HAT+ 26 TOPS (Hailo-8) | $109.99 | Micro Center | Ships or in-store pickup |
| 3 | USB-C PD Power Supply, 27W | $11.95 | PiShop.us | |
| 4 | 128GB microSD (SanDisk Extreme or similar A2) | ~$15.00 | Not re-verified this session | Original BOM.md estimate, commodity item |
| 5 | Active Cooler (recommended, not optional at sustained inference load) | ~$5.00 | raspberrypi.com | Prevents thermal throttling from skewing the FPS benchmark |

**POC total: ~$317**

## What's deliberately deferred

Everything in BOM.md's original Phase 0 list (camera, PoE injector) and all
of Phase 3+ (siren, relay, polarizer, outdoor cable, mount, surge
protector, case). None of it affects whether the Hailo-8 can hold 15 FPS
pose at 1080p, which is the only question P0 is answering.

## Buy trigger

Once P0 clears its exit criterion, buy in this order:

1. Camera (Reolink RLC-811A, $139.95) — needed for P1 real-footage capture
2. PoE injector (~$20) — needed to power the camera on the pool mount
3. Everything in BOM.md's Phase 3+ table — needed for the actual install

This keeps spend proportional to validated progress: nothing outdoor-rated
or install-specific gets bought until the compute path is proven.
