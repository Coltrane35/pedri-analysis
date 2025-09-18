#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
from collections import Counter, defaultdict
from typing import Dict, Any, Optional, Iterator

from utils.io_compat import setup_stdout_utf8, print_safe

setup_stdout_utf8()

PEDRI_ID = 30486
EVENTS_DIR = Path("data/events")


def iter_event_files() -> Iterator[Path]:
    if not EVENTS_DIR.exists():
        print_safe(f"⚠️  Katalog {EVENTS_DIR} nie istnieje.")
        return iter(())
    return (p for p in sorted(EVENTS_DIR.glob("*.json")) if p.is_file())


def load_events(path: Path) -> Optional[list[Dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as e:
        print_safe(f"⚠️  Nie udało się wczytać {path.name}: {e}")
    return None


def extract_match_id(events, fallback: str) -> str:
    try:
        mid = events[0].get("match_id")
        if mid is not None:
            return str(mid)
    except Exception:
        pass
    return Path(fallback).stem


def pedri_position_in_starting_xi(events) -> Optional[str]:
    for ev in events:
        if (ev.get("type") or {}).get("name") != "Starting XI":
            continue
        lineup = (ev.get("tactics") or {}).get("lineup") or ev.get("lineup") or []
        for pl in lineup:
            pid = pl.get("player_id") or (pl.get("player") or {}).get("id")
            if pid == PEDRI_ID:
                return (pl.get("position") or {}).get("name", "Unknown")
    return None


def main() -> None:
    print_safe("Analizuję pozycje Pedriego w Starting XI...")
    pos_counter = Counter()
    matches_by_pos = defaultdict(list)

    scanned = 0
    for path in iter_event_files():
        scanned += 1
        events = load_events(path)
        if not events:
            continue
        match_id = extract_match_id(events, path.name)
        pos = pedri_position_in_starting_xi(events)
        if pos:
            pos_counter[pos] += 1
            matches_by_pos[pos].append(match_id)

    print_safe("\n📊 Rozkład pozycji:")
    for pos, cnt in pos_counter.most_common():
        print_safe(f" - {pos}: {cnt}")

    print_safe("\n📄 Mecze per pozycja (maks 10 per pozycja dla podglądu):")
    for pos, mids in matches_by_pos.items():
        preview = ", ".join(mids[:10])
        more = f" (+{len(mids)-10})" if len(mids) > 10 else ""
        print_safe(f" - {pos}: {preview}{more}")

    print_safe(f"\n✅ Zakończono. Przeskanowano plików: {scanned}")


if __name__ == "__main__":
    main()
