# PoolGuard — AI Pool Drowning Detection System

**Status:** Draft v0.1 · 2026-07-24
**Owner:** Sean Enright

---

## 1. Problem Statement

Roughly 4,000 fatal unintentional drownings occur annually in the US. Pools and spas
account for ~350–400 deaths per year among children under 15, and pools represent
~21% of drownings across all demographics. Drowning is fast and quiet: a struggling
swimmer typically has 20–60 seconds on the surface, and irreversible brain damage
begins ~4 minutes after submersion. Most residential pools have **zero** active
monitoring — supervision lapses of under two minutes are implicated in the majority
of child pool drownings.

Commercial detection systems exist but are expensive and closed:

| Product | Approach | Price |
|---|---|---|
| Coral Manta 3000 / MYLO | Underwater camera + on-device AI, siren + app | ~$1,999 |
| AngelEye | Networked underwater cameras (commercial pools) | $10k+ installs |
| Poseidon | Overhead + underwater cameras (commercial pools) | $100k+ installs |
| Float/wristband alarms (e.g. Safety Turtle) | Immersion sensors, wearable | ~$150–200 |

**Gap:** no open, affordable (<$500), camera-based system for residential pools.

## 2. Goal

Build a self-hosted system — one camera plus an edge-AI computer — that watches a
residential outdoor pool 24/7 and:

1. Detects a person **struggling** at the surface (distress behavior).
2. Detects a person **motionless or submerged at the bottom** of the pool.
3. Detects **unsupervised entry** (e.g., a small child entering the pool area/water).
4. Fires layered alerts within seconds: local siren → phone push → smart home →
   SMS/call escalation if unacknowledged.

### Non-Goals (v1)

- Not a replacement for adult supervision, fencing, or door alarms. This is one
  layer in the CPSC "layers of protection" model, and all docs/UI must say so.
- No cloud inference — all detection runs on-device (privacy, latency, no
  internet dependency for the siren path).
- No certification as a life-safety device (UL 2621 etc.) in v1.
- Multi-pool / commercial deployments.

## 3. Users

- **Primary:** Sean's household — outdoor residential pool, family with guests/kids.
- **Secondary (open-source release):** technically-inclined pool owners who can
  mount a camera and flash an SD card.

## 4. Success Criteria

| Metric | Target |
|---|---|
| Detection-to-siren latency | < 5 s from sustained distress signal |
| Submersion detection | Alert within 15 s of a person motionless underwater |
| False alarms (siren-level) | < 1/week during normal swim activity |
| Uptime | 24/7, auto-recovery on power/network loss, watchdog + heartbeat alert if the system itself goes down |
| Night operation | Detection functional in dark via IR or supplemental lighting |
| Cost | ≤ $500 total hardware |

False-negative vs false-positive stance: **bias toward sensitivity.** A weekly
false siren is acceptable; a missed submersion is not. Escalation tiers (below)
let lower-confidence events fire softer alerts.

## 5. System Overview

```
[PoE Camera] --RTSP--> [Pi 5 + AI accelerator]
                          ├─ Person detection + pose (YOLO on Hailo NPU)
                          ├─ Multi-object tracking (ByteTrack)
                          ├─ Behavior rules engine (distress / submersion / entry)
                          └─ Alert manager
                               ├─ GPIO relay → 12V siren (no-network path)
                               ├─ Push (ntfy/Pushover) with snapshot
                               ├─ MQTT → Home Assistant (lights, speakers)
                               └─ Twilio SMS/voice escalation if unacknowledged
```

### Detection pipeline (v1)

1. **Person detection + pose estimation** — YOLO11-pose (or YOLOv8n-pose) compiled
   for the Hailo NPU, ~20–30 FPS at 1080p. Water-splash/glare robustness comes from
   fine-tuning on pool datasets (several exist on Roboflow; academic models like
   MS-YOLO / YOLO11-LiB report 90%+ drowning-class mAP).
2. **Tracking** — persistent IDs per swimmer so the rules engine reasons about
   *individuals over time*, not single frames.
3. **Rules engine** (interpretable, tunable — not end-to-end black box):
   - *Distress:* vertical posture + high-frequency arm motion + little horizontal
     travel, sustained > N seconds ("instinctive drowning response" signature).
   - *Submersion:* tracked person disappears below surface and does not resurface
     within N seconds; or detected shape at bottom with no motion.
   - *Unsupervised entry:* person detected in water when system is in "armed"
     mode (no adult present / scheduled quiet hours).
4. **Alert tiers:** `watch` (app note) → `warn` (push + HA announcement) →
   `emergency` (siren + push + escalation timer → SMS/call).

### Operating modes

- **Armed** (default/away/night): any entry into water = immediate emergency.
- **Swim mode** (manually or geofence-activated): entry is fine; distress and
  submersion rules active.
- **Maintenance:** all detection paused, auto-rearms after timeout.

## 6. Hardware (BOM, ~$460)

| Item | Choice | Est. |
|---|---|---|
| Compute | Raspberry Pi 5, 8 GB | $80 |
| AI accelerator | Raspberry Pi AI HAT+ 26 TOPS (Hailo-8) — or 13 TOPS Hailo-8L version at $70 | $110 |
| Camera | Reolink RLC-811A or Amcrest 4K PoE turret — IP67, RTSP, IR night vision, optical zoom to frame the pool | $110 |
| PoE | Single-port PoE+ injector (or small PoE switch) | $25 |
| Storage/PSU/case | 128 GB microSD (or NVMe HAT), 27 W USB-C PSU, cooled case | $60 |
| Siren | 12 V 110 dB siren + relay HAT + 12 V supply (hardwired = works with WiFi down) | $45 |
| Lens filter | Circular polarizer sized to camera lens (cuts surface glare) | $15 |
| Mounting | Pole/eave mount, outdoor-rated cable | $15 |

**Alternative compute:** NVIDIA Jetson Orin Nano Super dev kit ($249) — ~4× the
NPU headroom for bigger models, at the cost of ~$100 over the Pi path and more
power draw. Decision deferred; Pi 5 + Hailo is the default for cost and the
mature Frigate/rpicam ecosystem. The camera, siren, and alerting layers are
identical either way, so switching later is cheap.

**Placement:** camera mounted high (roof eave / 10 ft pole) looking down the long
axis of the pool at a steep angle — steep angles + polarizer minimize surface
glare and maximize through-water visibility. Pi lives indoors/garage; only the
PoE cable runs outside.

## 7. Above-Water vs Underwater Camera Analysis

| | Above-water (elevated) | Underwater |
|---|---|---|
| Surface distress detection | **Excellent** — full-body pose visible | Poor–fair (sees legs/torso from below) |
| Bottom-of-pool detection | Fair — depends on glare, ripple, depth, IR is useless through water at night | **Excellent** — clear silhouette against surface light |
| Entry detection | **Excellent** — sees whole pool area incl. deck | None (water only) |
| Night | Good with IR/lighting for surface; weak through water | Needs underwater illumination |
| Install | Screw a mount, run one cable | Through-wall penetration, or over-the-edge "periscope" (Coral Manta style), fouling/chemistry exposure |
| Cost | ~$110 | +$150–300 (housed cam or periscope rig + illumination) |
| Failure modes | Glare, rain on dome, heavy ripple | Biofouling, condensation, cable damage |

**Decision:** v1 ships above-water only — it covers distress and entry (the two
scenarios with the largest time-to-intervene windows) and *attempts* bottom
detection via track-loss logic ("person went under and never came back up"),
which doesn't require actually seeing the bottom. **v2 adds an underwater view**
dedicated to bottom-of-pool confirmation, closing the main above-water blind
spot. This mirrors how Poseidon combines both views.

Two candidate form factors for the v2 underwater camera:

1. **Over-the-edge housed camera** — USB/IP cam in a dive housing on a weighted
   bracket (no wall penetration). Wired power/data, fixed viewpoint. Downside:
   visible rig on the pool edge, one cable into the water.
2. **Floating buoy ("chlorine floater" style)** — downward camera just below the
   waterline; this is Coral MYLO's form factor, so it's commercially validated.
   Zero install and excellent bottom visibility, but: battery/solar power budget
   forces on-float inference with event-only transmission, IP68 sealing and
   chlorine fouling are nontrivial, the viewpoint drifts/rotates, and it will be
   treated as a pool toy. Explicitly **not viable as the sole camera** — it
   cannot see the deck (no entry detection) and water-level views of surface
   distress are poor.

## 8. Software Stack

- **OS/base:** Raspberry Pi OS Lite, Docker Compose for services.
- **Inference:** Hailo runtime + Ultralytics-exported model; Frigate NVR is the
  fallback/companion for recording and motion gating (build vs. adopt decision
  in open questions).
- **Language:** Python for the vision pipeline (ecosystem lock-in); typed with
  Pydantic models for all events/config.
- **Alerting:** ntfy or Pushover for `watch`/`warn` tiers (unlimited, chatty),
  MQTT → Home Assistant (siren backup path, lights, speaker announcements), and
  **PagerDuty Free** for the `emergency` tier — its Events API + escalation
  policy replaces a hand-rolled Twilio ack/retry ladder, and its mobile app's
  critical-alert entitlement can break through a muted/DND phone, which ntfy
  cannot. Free plan: 5 users, 1 escalation policy, unlimited push, 100 SMS/mo
  (only unacked emergencies consume quota). Twilio remains the fallback if the
  free tier's phone-call support disappoints; Squadcast/Zenduty free tiers and
  self-hosted GoAlert are the alternates.
- **Config/UI:** YAML config + simple local web page for mode switching, live
  view, alert acknowledgment, and event review clips. HA dashboard covers most
  of this early on.
- **Data:** every alert saves a 30 s clip locally for tuning; opt-in labeled
  dataset grows the fine-tuning corpus.

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Missed detection → false sense of security | Critical | Sensitivity-biased tuning; prominent "supplement, not substitute" framing; heartbeat/self-test alert when system is degraded (camera offline, low FPS, lens blocked) |
| Glare/ripple/night degrade accuracy | High | Steep mount angle, polarizer, IR + optional landscape lighting, fine-tune on own footage, v2 underwater cam |
| False alarms cause alert fatigue → siren gets disabled | High | Tiered alerts; only high-confidence sustained events reach siren; weekly false-alarm budget tracked as a metric |
| Toddler-sized subjects underrepresented in datasets | High | Explicit test protocol with mannequin/doll of child size; collect + label own footage |
| Pi/NPU can't sustain FPS with pose model | Medium | Hailo-8 26 TOPS headroom; fall back to detection-only + track heuristics; Jetson escape hatch |
| Outdoor cam/water damage | Medium | IP67 gear, PoE surge protector |
| Privacy (cameras + swimsuits + guests) | Medium | Local-only processing/storage, no cloud, clips auto-expire, visible signage for guests |

## 10. Phases

1. **P0 — Bench rig:** Pi + AI HAT + camera on desk; run pretrained YOLO pose on
   RTSP; measure FPS/latency. *Exit: 15+ FPS pose at 1080p.*
2. **P1 — Pool install + recording:** mount camera, capture real footage across
   sun/rain/night/swimmers; build the labeled clip library. *Exit: 2 weeks of
   representative footage.*
3. **P2 — Detection + rules:** tracking + distress/submersion/entry rules; replay
   harness that runs the pipeline against recorded clips as the test suite.
   *Exit: rules fire correctly on staged scenarios (adult mock-distress,
   weighted mannequin sink test, child-doll entry).*
4. **P3 — Alerting:** siren relay, push, HA, Twilio escalation, ack flow, armed/
   swim modes. *Exit: end-to-end staged drill under 5 s to siren.*
5. **P4 — Hardening:** watchdogs, self-test heartbeat, false-alarm tuning over a
   month of live use.
6. **P5 (v2) — Underwater camera** + open-source release (docs, install guide).

## 11. Open Questions

- Build the pipeline custom vs. extend Frigate (Frigate gives recording, motion
  gating, HA integration for free, but its detector/tracking hooks may be too
  rigid for pose-based distress rules). Leaning: custom pipeline, Frigate
  alongside purely as NVR.
- Which public drowning datasets are license-compatible and actually transfer to
  an elevated-outdoor-residential viewpoint? (Most academic sets are indoor
  commercial pools.)
- Hailo-8 (26 TOPS, $110) vs Hailo-8L (13 TOPS, $70): benchmark pose model on
  both before buying, if possible.
- Geofence/presence-based auto-arming vs manual-only modes for v1.
- v2 underwater form factor: over-the-edge housed cam vs floating buoy (PRD §7).
  A Pi Zero 2 W + camera in a sealed dry-box on a foam collar is a cheap
  weekend prototype to de-risk the buoy option.
- Verify PagerDuty Free actually includes phone-call notifications (sources
  conflict: SMS-only vs 100 phone+SMS/mo) — requires signing up and testing an
  escalation. If calls are paid-only, evaluate Squadcast free tier next.
- Legal review before public release: liability language for a safety-adjacent
  open-source project (disclaimer, no warranty, "not a life-saving device").

## 12. References

- [Coral Manta 3000 (Amazon)](https://www.amazon.com/Coral-Manta-3000-Pool-Alarm/dp/B0861MNZ95) · [In The Swim listing](https://www.intheswim.com/p/coral-manta-3000-drowning-detection-system) · [MYLO](https://coralmylo.com/)
- [Drowning detection systems overview (Wikipedia)](https://en.wikipedia.org/wiki/Drowning_detection_system)
- [YOLO11-LiB drowning model paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12431139/) · [MS-YOLO paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11548417/)
- [Automated-Drowning-Detection-YOLOV8 (GitHub)](https://github.com/Hasibwajid/Automated-Drowning-Detection-YOLOV8) · [drowning-detection topic](https://github.com/topics/drowning-detection) · [Above+underwater YOLO system](https://github.com/zseng0912/Drowning-Detection-System)
- [DIY swim-safety build writeup (Hackers Vanguard)](https://hackersvanguard.com/ai-swim-safety-monitoring/)
- [Raspberry Pi AI Kit](https://www.raspberrypi.com/products/ai-kit/) · [Pi 5 + Hailo YOLO tutorial (Seeed)](https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/) · [Hailo setup guide](https://datarootlabs.com/blog/hailo-ai-kit-raspberry-pi-5-setup-and-computer-vision-pipelines)
- [Frigate YOLO detector plugin](https://github.com/dbro/frigate-detector-edgetpu-yolo9)
