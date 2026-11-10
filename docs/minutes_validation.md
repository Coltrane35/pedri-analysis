# P1 — Minutes and per-90 validation

Status: **in progress**. This report does not claim that ten real matches have been verified.

## Method
- Identify player in `Starting XI` or as `Substitution.replacement`.
- Entry is 0 for starters, or substitution event clock for substitutes.
- Exit is substitution event clock, or the last `Half End` event clock.
- Preserve stoppage time and extra time; no artificial 95-minute cap.
- Treat missing `Half End`, missing player-entry evidence, or contradictory events as invalid (`None`).
- Per-90 uses `90 * count / estimated_minutes`; invalid minutes produce `NaN`.

**Limitation:** Event timestamps are an approximation of elapsed playing time, not official playing minutes. Some providers normalize player minutes differently. Validate against match timeline and document differences.

## Ten-match review checklist

| Match ID | Existing estimate | Verified start/entry | Verified exit/end | Recomputed | Difference | Evidence |
|---|---:|---|---|---:|---:|---|
| 3942226 | 7.7 | TBD | TBD | TBD | TBD | event JSON |
| 3773466 | 23.9 | TBD | TBD | TBD | TBD | event JSON |
| 3773386 | 50.0 | TBD | TBD | TBD | TBD | event JSON |
| 3773523 | 73.9 | TBD | TBD | TBD | TBD | event JSON |
| 3773457 | 45.0 | TBD | TBD | TBD | TBD | event JSON |
| 3773369 | 84.5 | TBD | TBD | TBD | TBD | event JSON |
| 3764440 | 93.0 | TBD | TBD | TBD | TBD | event JSON |
| 3794686 | 121.1 | TBD | TBD | TBD | TBD | event JSON |
| 3795220 | 126.0 | TBD | TBD | TBD | TBD | event JSON |
| 3857291 | 56.5 | TBD | TBD | TBD | TBD | event JSON |

## Regression coverage
Synthetic fixtures test starter, substitute entry, early exit, stoppage time, extra time, missing match end, contradictory substitutions, and invalid /90 denominator.

## Acceptance criteria
- [ ] 10 real event files reviewed and differences explained
- [x] Regression tests written
- [ ] Regression tests run in repository environment
- [ ] Existing analysis pipeline updated and CSV outputs regenerated
- [ ] GitHub commit and evidence links recorded
