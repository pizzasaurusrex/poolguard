# Bill of Materials

Prices checked 2026-07-24. Phase 0 items are all you need to start —
bench-test FPS before buying/mounting the rest.

## Phase 0 — bench rig (~$300)

| # | Item | Link | Est. |
|---|---|---|---|
| 1 | Raspberry Pi 5, 8 GB | [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-5/) · [PiShop.us](https://www.pishop.us/product/raspberry-pi-5-8gb/) | $80 |
| 2 | Raspberry Pi AI HAT+ 26 TOPS (Hailo-8) | [raspberrypi.com](https://www.raspberrypi.com/products/ai-hat/) · [PiShop.us](https://www.pishop.us/product/raspberry-pi-ai-hat-26-tops/) · [SparkFun](https://www.sparkfun.com/raspberry-pi-ai-hat-26-tops.html) | $110 |
| 3 | Reolink RLC-811A 4K PoE camera (IP67, RTSP, 5× optical zoom) | [reolink.com](https://reolink.com/product/rlc-811a/) · [Amazon](https://www.amazon.com/REOLINK-Security-Detection-Spotlight-Time-Lapse/dp/B09873G7X3) | $110 |
| 4 | 27 W USB-C PSU (official) | [raspberrypi.com](https://www.raspberrypi.com/products/27w-power-supply/) | $12 |
| 5 | 128 GB microSD (SanDisk Extreme or similar A2 card) | [Amazon](https://www.amazon.com/dp/B09X7C7LL1) | $15 |
| 6 | PoE+ injector, 802.3at (TP-Link TL-PoE160S) | [Amazon](https://www.amazon.com/dp/B01MDLUSE7) | $20 |

Buy #1–2 from the same Pi retailer (PiShop.us / SparkFun / CanaKit) to save
shipping. The camera ships with a short Cat5 lead; any Ethernet cable to your
router works for the bench.

Notes:
- **13 vs 26 TOPS (open question, PRD §11):** buying the 26 TOPS ($110 vs $70)
  is the recommended default — pose models + headroom for a second camera in
  v2. If P0 benchmarks show huge slack, nothing is lost; if 13 TOPS came up
  short, we'd be re-buying.
- The AI HAT+ occupies the Pi's single PCIe lane, so storage stays on microSD
  (fine — video recording goes to USB SSD if needed later).
- Active cooling: the official case fan or Active Cooler ($5,
  [raspberrypi.com](https://www.raspberrypi.com/products/active-cooler/)) —
  cheap insurance for sustained inference. The AI HAT+ needs the 16mm GPIO
  stacker (included) above the cooler.

## Phase 3+ — pool install (~$160, buy after P0 passes)

| # | Item | Link | Est. |
|---|---|---|---|
| 7 | 12 V 110 dB motor siren | [Amazon search: "12V 110dB siren alarm"](https://www.amazon.com/s?k=12v+110db+siren+alarm) | $15 |
| 8 | Relay HAT / 5 V relay module (opto-isolated) | [Amazon search: "5V relay module optocoupler"](https://www.amazon.com/s?k=5v+relay+module+optocoupler) | $10 |
| 9 | 12 V 2 A power supply for siren | [Amazon search](https://www.amazon.com/s?k=12v+2a+power+supply) | $10 |
| 10 | Circular polarizer sized to the RLC-811A lens (measure bezel; often a 62–67 mm CPL + adhesive mount, or foil-style CPL sheet) | [Amazon search: "CPL filter security camera"](https://www.amazon.com/s?k=cpl+filter+security+camera) | $15 |
| 11 | Outdoor Cat6 (direct-burial rated), length per site survey | [Amazon search](https://www.amazon.com/s?k=outdoor+cat6+direct+burial) | $25 |
| 12 | Pole/eave camera mount + stainless hardware | [Amazon search: "security camera pole mount"](https://www.amazon.com/s?k=security+camera+pole+mount) | $25 |
| 13 | Ethernet surge protector (camera side) | [Amazon search: "PoE ethernet surge protector"](https://www.amazon.com/s?k=poe+ethernet+surge+protector) | $15 |
| 14 | Pi case (cooled, HAT clearance) or vented enclosure in garage | [Amazon search: "Raspberry Pi 5 case AI HAT"](https://www.amazon.com/s?k=raspberry+pi+5+case+ai+hat) | $20 |

**Total both phases: ~$460** — within the ≤$500 budget (PRD §4).
Items 7–14 are commodity parts; search links given rather than specific
listings since stock churns. Pick the siren/relay after deciding indoor vs
outdoor siren placement.
