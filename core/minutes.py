"""Event-clock playing-time estimates for StatsBomb open data.

Minutes are estimates from event timestamps, not official minutes.
"""
from __future__ import annotations

import math


def name(event):
    return (event.get('type') or {}).get('name')


def clock(event):
    try:
        minute = float(event['minute'])
        second = float(event.get('second', 0))
        value = minute + second / 60
        if not math.isfinite(value) or value < 0:
            return None
        return value
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def estimate_minutes(events, player_id=30486):
    """Return (minutes|None, details). None signals insufficient/inconsistent evidence.

    Uses global match clock in StatsBomb minute/second. No 95-minute cap.
    """
    starts = any(
        name(e) == 'Starting XI' and any(
            (p.get('player') or {}).get('id') == player_id
            for p in ((e.get('tactics') or {}).get('lineup') or [])
        ) for e in events
    )
    incoming = []
    outgoing = []
    for e in events:
        if name(e) != 'Substitution':
            continue
        t = clock(e)
        if t is None:
            continue
        if ((e.get('substitution') or {}).get('replacement') or {}).get('id') == player_id:
            incoming.append(t)
        if (e.get('player') or {}).get('id') == player_id:
            outgoing.append(t)

    ends = [clock(e) for e in events if name(e) == 'Half End']
    ends = [t for t in ends if t is not None]
    if not ends:
        return None, {'reason': 'missing_half_end', 'starts': starts}
    end = max(ends)
    if starts and incoming:
        return None, {'reason': 'conflicting_start_and_entry', 'starts': starts}
    if len(incoming) > 1 or len(outgoing) > 1:
        return None, {'reason': 'multiple_substitutions', 'starts': starts}
    if not starts and not incoming:
        return None, {'reason': 'no_start_or_entry', 'starts': starts}
    start = 0.0 if starts else incoming[0]
    stop = outgoing[0] if outgoing else end
    if stop < start or stop > end + 0.25:
        return None, {'reason': 'invalid_interval', 'start': start, 'stop': stop, 'match_end': end}
    return round(stop - start, 3), {
        'reason': 'estimated', 'starts': starts, 'start': start,
        'stop': stop, 'match_end': end, 'entered': bool(incoming), 'exited': bool(outgoing)
    }


def per90(value, minutes):
    """Missing/invalid denominator yields NaN, never a misleading zero."""
    if minutes is None or not math.isfinite(float(minutes)) or minutes <= 0:
        return float('nan')
    return float(value) * 90.0 / float(minutes)
