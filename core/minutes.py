"""Event-clock playing-time estimates for StatsBomb open data.

Minutes are estimates from event timestamps, not official minutes.
"""

from __future__ import annotations

import math


def name(event):
    return (event.get("type") or {}).get("name")


def clock(event):
    try:
        minute = float(event["minute"])
        second = float(event.get("second", 0))

        value = minute + second / 60

        if not math.isfinite(value) or value < 0:
            return None

        return value

    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def estimate_minutes(events, player_id=30486):
    """Return (minutes | None, details).

    None means that the event timeline is insufficient or inconsistent.

    Rules:
    - starter -> starts at 0.0
    - substitute -> starts at substitution entry time
    - substituted player -> stops at substitution time
    - otherwise stops at the final Half End of periods 1-4
    - period 5 (penalty shootout) is ignored
    - no artificial 95-minute cap
    """

    starts = any(
        name(event) == "Starting XI"
        and any(
            (player.get("player") or {}).get("id") == player_id
            for player in ((event.get("tactics") or {}).get("lineup") or [])
        )
        for event in events
    )

    incoming = []
    outgoing = []

    for event in events:
        if name(event) != "Substitution":
            continue

        event_time = clock(event)

        if event_time is None:
            continue

        replacement_id = (
            ((event.get("substitution") or {}).get("replacement") or {}).get("id")
        )

        outgoing_id = (event.get("player") or {}).get("id")

        if replacement_id == player_id:
            incoming.append(event_time)

        if outgoing_id == player_id:
            outgoing.append(event_time)

    # Period 5 is a penalty shootout and must not count
    # towards playing time.
    ends = [
        clock(event)
        for event in events
        if name(event) == "Half End"
        and event.get("period") in (1, 2, 3, 4)
    ]

    ends = [event_time for event_time in ends if event_time is not None]

    if not ends:
        return None, {
            "reason": "missing_half_end",
            "starts": starts,
        }

    match_end = max(ends)

    if starts and incoming:
        return None, {
            "reason": "conflicting_start_and_entry",
            "starts": starts,
        }

    if len(incoming) > 1 or len(outgoing) > 1:
        return None, {
            "reason": "multiple_substitutions",
            "starts": starts,
        }

    if not starts and not incoming:
        return None, {
            "reason": "no_start_or_entry",
            "starts": starts,
        }

    start_time = 0.0 if starts else incoming[0]
    stop_time = outgoing[0] if outgoing else match_end

    if stop_time < start_time or stop_time > match_end + 0.25:
        return None, {
            "reason": "invalid_interval",
            "start": start_time,
            "stop": stop_time,
            "match_end": match_end,
        }

    minutes = round(stop_time - start_time, 3)

    return minutes, {
        "reason": "estimated",
        "starts": starts,
        "start": start_time,
        "stop": stop_time,
        "match_end": match_end,
        "entered": bool(incoming),
        "exited": bool(outgoing),
    }


def per90(value, minutes):
    """Calculate per-90 metric.

    Missing, zero or invalid minutes return NaN instead of
    a misleading zero.
    """

    if minutes is None:
        return float("nan")

    try:
        minutes = float(minutes)
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if not math.isfinite(minutes) or minutes <= 0:
        return float("nan")

    if not math.isfinite(value):
        return float("nan")

    return value * 90.0 / minutes