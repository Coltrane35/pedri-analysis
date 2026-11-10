import math
from core.minutes import estimate_minutes, per90

PID = 30486

def start():
    return {'type': {'name': 'Starting XI'}, 'tactics': {'lineup': [{'player': {'id': PID}}]}}

def half_end(m):
    return {'type': {'name': 'Half End'}, 'minute': m, 'second': 0}

def sub_out(m):
    return {'type': {'name': 'Substitution'}, 'player': {'id': PID}, 'minute': m, 'second': 0,
            'substitution': {'replacement': {'id': 999}}}

def sub_in(m):
    return {'type': {'name': 'Substitution'}, 'player': {'id': 999}, 'minute': m, 'second': 0,
            'substitution': {'replacement': {'id': PID}}}

def test_full_match_with_stoppage():
    minutes, _ = estimate_minutes([start(), half_end(46), half_end(94)])
    assert minutes == 94

def test_bench_entry():
    minutes, details = estimate_minutes([sub_in(65), half_end(46), half_end(94)])
    assert minutes == 29 and details['entered']

def test_early_exit():
    minutes, details = estimate_minutes([start(), sub_out(73), half_end(46), half_end(94)])
    assert minutes == 73 and details['exited']

def test_extra_time_not_capped():
    minutes, _ = estimate_minutes([start(), half_end(46), half_end(94), half_end(107), half_end(123)])
    assert minutes == 123

def test_missing_end_is_invalid():
    minutes, details = estimate_minutes([start()])
    assert minutes is None and details['reason'] == 'missing_half_end'

def test_invalid_interval_is_invalid():
    minutes, details = estimate_minutes([sub_in(80), sub_out(60), half_end(94)])
    assert minutes is None and details['reason'] == 'invalid_interval'

def test_per90():
    assert per90(10, 45) == 20
    assert math.isnan(per90(10, 0))
    assert math.isnan(per90(10, None))
