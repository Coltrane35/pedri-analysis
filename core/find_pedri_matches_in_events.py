#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, Optional, Iterator, Set

from utils.io_compat import setup_stdout_utf8, print_safe

setup_stdout_utf8()

PEDRI_ID = 30486
EVENTS_DIR = Path("data/events")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def pedri_present_in_any_event(events) -> bool:
    for ev in events:
        player = ev.get("player") or {}
        pid = player.get("id")
        if pid == PEDRI_ID:
            return True
    return False


def main() -> None:
    print_safe(
        "Szukam wszystkich meczów, w których Pedri wystąpił (jakikolwiek event Pedriego)..."
    )
    matches: Set[str] = set()
    scanned = 0

    for path in iter_event_files():
        scanned += 1
        events = load_events(path)
        if not events:
            continue
        match_id = extract_match_id(events, path.name)
        if pedri_present_in_any_event(events):
            matches.add(match_id)
            print_safe(f"✅ {match_id}")

    out_path = OUT_DIR / "pedri_match_ids.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for mid in sorted(matches):
            f.write(f"{mid}\n")

    print_safe(f"\n🔢 Znaleziono {len(matches)} meczów z udziałem Pedriego.")
    print_safe(f"💾 Zapisano listę do: {out_path}")
    print_safe(f"Przeskanowano plików: {scanned}")


if __name__ == "__main__":
    main()
