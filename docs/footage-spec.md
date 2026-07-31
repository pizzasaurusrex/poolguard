# Footage spec — self-shot POC clips

Derived from the first tracker run over the found-footage library (2026-07-31).
Found clips are cut for drama: every one starts mid-event, which is why four of
seven positives cannot exercise their rule and the submersion clip cannot
produce a track at all.

> **Safety first.** Nobody simulates drowning unsupervised. Use a competent
> swimmer, a spotter on the deck whose only job is watching, and a rehearsed
> stop signal. For submersion takes, breath-hold voluntarily — never weighted,
> never restrained. A mannequin or weighted doll is the safer instrument for
> anything longer than a comfortable breath-hold.

## Why lead-in is mandatory

Every rule is a threshold over a *track history*, and the clock starts when the
behavior starts. Two consequences:

- A clip can only test its rule if `duration > lead_in + threshold + margin`.
- Submersion works by a track existing and then coasting. If the person is
  already under at frame one, no track ever spawns, so there is nothing to
  lose. **A submersion clip must show the swimmer above water first.**

At current thresholds (`distress_sustained_seconds` 5.0 s,
`submersion_timeout_seconds` 12.0 s):

| clip type   | lead-in | event runway | total     |
|-------------|---------|--------------|-----------|
| distress    | 10 s    | 10 s         | ~20–30 s  |
| submersion  | 10 s    | 20 s         | ~30–45 s  |
| negatives   | —       | —            | 30–60 s   |

## Camera

The tracker assumes a fixed camera — there is no motion model, and association
is IoU between frames. Handheld footage will fragment tracks for reasons the
Pi-mounted camera never will, so it tests the wrong thing.

- Tripod or hard mount. No panning, no zoom, no handheld.
- Mount high and angled down, approximating the eventual install: the whole
  water surface in frame, minimal sky.
- Shoot the same scene in different light — midday glare and low sun are where
  detection collapsed in the found footage.
- One continuous take per scenario. No cuts, so no splitting is needed.

## Shot list

**Positives**

1. *Distress, single adult* — normal swimming, then vertical struggle, head low,
   arms pressing down, no forward progress. Hold 10 s+.
2. *Submersion, single adult* — visible swimming, then slips under and stays
   under 20 s+ while the camera keeps rolling. The single most valuable clip.
3. *Child falls in* — entry from the deck, then struggle. Use a doll or
   mannequin for the entry if a real child is involved in any way.
4. *Distress with bystanders* — one person in trouble while others swim
   normally. Tests that the rule fires per-track, not per-frame.

**Negatives — equally important, this is the false-alarm budget**

5. *Repeated breath-hold dives* — under for 8–11 s, surface, repeat. This is the
   nearest miss to submersion and the likeliest false siren.
6. *Treading water, chatting* — vertical, stationary, arms moving. Visually the
   closest thing to distress.
7. *Vigorous normal play* — splashing, jumping in, horseplay.
8. *Empty pool* — surface glare, moving shadows, floats drifting. Must produce
   zero tracks.

## Labeling

Add each to `tests/labels/shots.yaml` after shooting. Record the wall-clock
second the behavior starts — that is what makes `onset_window` a real assertion
rather than a guess. Set `onset_window` to
`[behavior_start + threshold, behavior_start + threshold + margin]`.
