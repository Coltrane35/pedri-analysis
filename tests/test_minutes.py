import math

from core.minutes import estimate_minutes, per90


PID = 30486


def start():
    return {
        "type": {"name": "Starting XI"},
        "tactics": {
            "lineup": [
                {
                    "player": {"id": PID},
                }
            ]
        },
    }


def half_end(minute, period, second=0):
    return {
        "type": {"name": "Half End"},
        "period": period,
        "minute": minute,
        "second": second,
    }


def sub_out(minute):
    return {
        "type": {"name": "Substitution"},
        "player": {"id": PID},
        "minute": minute,
        "second": 0,
        "substitution": {
            "replacement": {"id": 999},
        },
    }


def sub_in(minute):
    return {
        "type": {"name": "Substitution"},
        "player": {"id": 999},
        "minute": minute,
        "second": 0,
        "substitution": {
            "replacement": {"id": PID},
        },
    }


def test_full_match_with_stoppage():
    events = [
        start(),
        half_end(46, period=1),
        half_end(94, period=2),
    ]

    minutes, details = estimate_minutes(events)

    assert minutes == 94
    assert details["starts"] is True
    assert details["entered"] is False
    assert details["exited"] is False


def test_bench_entry():
    events = [
        half_end(46, period=1),
        sub_in(65),
        half_end(94, period=2),
    ]

    minutes, details = estimate_minutes(events)

    assert minutes == 29
    assert details["entered"] is True
    assert details["exited"] is False


def test_early_exit():
    events = [
        start(),
        half_end(46, period=1),
        sub_out(73),
        half_end(94, period=2),
    ]

    minutes, details = estimate_minutes(events)

    assert minutes == 73
    assert details["exited"] is True


def test_extra_time_not_capped():
    events = [
        start(),
        half_end(46, period=1),
        half_end(94, period=2),
        half_end(107, period=3),
        half_end(123, period=4),
    ]

    minutes, details = estimate_minutes(events)

    assert minutes == 123
    assert details["match_end"] == 123


def test_penalty_shootout_not_counted_as_playing_time():
    events = [
        start(),
        half_end(44, period=1, second=53),
        half_end(93, period=2, second=5),
        half_end(105, period=3, second=8),
        half_end(120, period=4, second=4),
        half_end(126, period=5, second=3),
    ]

    minutes, details = estimate_minutes(events)

    assert minutes == 120.067
    assert round(details["match_end"], 3) == 120.067


def test_missing_end_is_invalid():
    minutes, details = estimate_minutes([start()])

    assert minutes is None
    assert details["reason"] == "missing_half_end"


def test_invalid_interval_is_invalid():
    events = [
        sub_in(80),
        sub_out(60),
        half_end(94, period=2),
    ]

    minutes, details = estimate_minutes(events)

    assert minutes is None
    assert details["reason"] == "invalid_interval"


def test_per90():
    assert per90(10, 45) == 20
    assert math.isnan(per90(10, 0))
    assert math.isnan(per90(10, None))