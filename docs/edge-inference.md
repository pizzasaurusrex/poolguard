# Edge Inference: AI HAT+ and the Model Strategy

How PoolGuard runs neural networks on-device. Companion to [PRD §5](../PRD.md#5-system-overview)–[§6](../PRD.md#6-hardware).

## The hardware

The Raspberry Pi AI HAT+ (26 TOPS, $110 MSRP) carries a **Hailo-8 NPU**
connected over the Pi 5's single PCIe Gen 3 lane. The 13 TOPS variant ($70)
uses the smaller Hailo-8L; we buy the 26 TOPS version — pose models are
heavier than plain detection, and the headroom covers a second camera stream
(v2 underwater) on the same chip.

The Hailo-8 is a **dataflow accelerator**, not a small GPU: the compiler maps
a network's layers onto the chip's compute fabric ahead of time. That's why it
delivers high throughput at ~2–5 W, and it drives the whole workflow:

- Models are **compiled, not loaded**. Trained model → ONNX → Hailo Dataflow
  Compiler (INT8 quantization) → `.hef` binary → executed by HailoRT.
  PyTorch checkpoints never run on the NPU directly.
- The **Hailo Model Zoo ships precompiled `.hef` files** for the models we
  need (YOLOv8/YOLO11 detection and pose). No compiler work required until we
  fine-tune.
- Raspberry Pi OS integrates the runtime: `sudo apt install hailo-all`.
  Hailo's [`hailo-rpi5-examples`](https://github.com/hailo-ai/hailo-rpi5-examples)
  repo has a working GStreamer pipeline (RTSP in → pose out) that P0 starts from.

## Division of labor

| Work | Runs on |
|---|---|
| Person detection + pose estimation | Hailo NPU |
| Video decode | Pi CPU (HEVC has HW assist; H.264 is software-only on Pi 5) |
| ByteTrack tracking, rules engine, alerting | Pi CPU (our Python code) |

Consequences:

- **Configure the camera for H.265/HEVC output.** The Pi 5 dropped hardware
  H.264 decode. Software H.264 works but eats CPU we'd rather spend on
  tracking. The P0 benchmark measures both if in doubt.
- The HAT occupies the only PCIe lane, so **no NVMe SSD** — OS on microSD,
  footage recording to microSD/USB.
- Cooling: use the active cooler; the HAT stacks above it on the 16 mm GPIO
  extender included in the box.

## Dev backend (no Hailo required)

Per ADR-010, the pipeline develops against a `PoseEstimator` seam with an
**Ultralytics backend** on the dev machine: YOLOv8/11-pose via the `vision`
extra, MPS-accelerated on Apple Silicon, emitting the same COCO 17-keypoint
`Detection` events the Hailo backend will. Differences to remember when
results move to the Pi:

- Dev runs FP16/FP32 weights; the `.hef` is INT8-quantized — expect slightly
  noisier keypoints on-target, and re-tune rule thresholds (they're config).
- Dev throughput says nothing about Hailo throughput; the 15 FPS exit
  criterion is only answerable on the bench rig.

## Model strategy (three stages)

1. **P0 — stock model.** Pretrained YOLOv8s-pose from the Hailo Model Zoo,
   COCO-trained: person boxes + 17 body keypoints. It knows nothing about
   drowning; P0 only proves ≥15 FPS end-to-end at 1080p.
2. **P2 — semantics in rules, not weights.** The drowning-specific
   intelligence lives in the interpretable rules engine (vertical posture +
   high-frequency arm motion + no horizontal travel = distress; track lost
   below surface + timeout = submersion). The model stays a generic OSS pose
   estimator. This was a deliberate PRD decision — rules are tunable and
   auditable; a black-box drowning classifier is neither.
3. **P4/v2 — fine-tune.** Retrain on our labeled pool footage (glare, ripple,
   night IR, toddler-sized subjects), optionally mixing academic drowning
   datasets, then recompile to `.hef` via Hailo's retraining docs.

## Licensing (matters for the P5 public release)

Ultralytics YOLOv8/11 **code and weights are AGPL-3.0** — redistributing them
inside an MIT-licensed release is a conflict. Options, decided before P5
([PRD §11](../PRD.md#11-open-questions)):

- Treat the model as an external artifact users fetch from the Hailo Model
  Zoo at install time (common, defensible), or
- Switch to a permissively licensed pose model: RTMPose or YOLOX
  (Apache-2.0), both with Hailo-compatible variants.

Private use on our own hardware is unaffected either way.
