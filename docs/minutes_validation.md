# P1 — Minutes and per-90 validation

Status: **completed**

## Method

Playing time is estimated from the StatsBomb event timeline:

- A player in `Starting XI` starts at minute `0.0`.
- A substitute starts at the timestamp of the substitution event where the player appears as `Substitution.replacement`.
- A substituted player stops at the timestamp of their substitution event.
- A player who remains on the pitch stops at the final `Half End` event from periods 1–4.
- Stoppage time and extra time are preserved.
- Period 5 is excluded because it represents a penalty shootout and does not count as playing time.
- There is no artificial 95-minute cap.
- Missing or contradictory timeline evidence produces `None`.
- Per-90 metrics use `90 * count / estimated_minutes`.
- Invalid minutes produce `NaN` rather than a misleading zero.

**Limitation:** these values are derived from event timestamps and represent estimated elapsed playing time rather than official provider-reported player minutes.

## Ten-match validation

| Match ID | Previous estimate | Verified appearance | Verified exit/end | Recomputed | Difference | Result |
|---|---:|---|---|---:|---:|---|
| 3942226 | 7.7 | Starter | substituted 7.717 | 7.717 | +0.017 | OK |
| 3773466 | 23.9 | entered 71.233 | match end 95.117 | 23.883 | -0.017 | OK |
| 3773386 | 50.0 | entered 45.000 | match end 94.950 | 49.950 | -0.050 | OK |
| 3773523 | 73.9 | Starter | substituted 73.900 | 73.900 | 0.000 | OK |
| 3773457 | 45.0 | Starter | substituted 45.000 | 45.000 | 0.000 | OK |
| 3773369 | 84.5 | Starter | substituted 84.483 | 84.483 | -0.017 | OK |
| 3764440 | 93.0 | Starter | match end 92.983 | 92.983 | -0.017 | OK |
| 3794686 | 121.1 | Starter | extra-time end 121.083 | 121.083 | -0.017 | OK |
| 3795220 | 126.0 | Starter | extra-time end 120.067 | 120.067 | -5.933 | **Fixed: shootout excluded** |
| 3857291 | 56.5 | Starter | substituted 56.517 | 56.517 | +0.017 | OK |

## Key finding

Nine of the ten reviewed matches were consistent with the expected event timeline apart from minor rounding differences.

Match `3795220` exposed an important edge case. The event data contained:

- period 4 `Half End`: `120:04`
- period 5 `Half End`: `126:03`

The previous implementation used the final `Half End` event without distinguishing the event period. As a result, the penalty shootout was incorrectly included in Pedri's playing time and produced approximately `126.05` minutes.

The implementation was changed so that only `Half End` events from periods 1–4 can define the end of playing time. Period 5 is explicitly excluded.

The corrected value for match `3795220` is:

`120.067 minutes`

instead of:

`126.050 minutes`

This also corrects all per-90 metrics for that match because the denominator is no longer inflated by the penalty shootout.

## Regression coverage

Regression tests cover:

- full-match starter,
- stoppage time,
- substitute entry,
- early substitution,
- extra time,
- penalty shootout exclusion,
- missing match end,
- contradictory substitution interval,
- zero or missing per-90 denominator.

The penalty-shootout case is now explicitly protected by:

`test_penalty_shootout_not_counted_as_playing_time`

## Acceptance criteria

- [x] 10 real event files reviewed and differences explained
- [x] Regression tests written
- [x] Regression tests run in repository environment
- [x] Existing analysis pipeline updated and CSV outputs regenerated
- [x] Penalty shootout edge case identified and fixed