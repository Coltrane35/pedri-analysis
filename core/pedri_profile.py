# pedri_profile.py
# -*- coding: utf-8 -*-
import os, json, glob
from collections import Counter

PEDRI_ID = 30486
EVENTS_DIR = os.path.join("data", "events")


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  {path}: {e}")
        return []


def main():
    files = sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json")))
    files += sorted(glob.glob(os.path.join(EVENTS_DIR, "**", "*.json"), recursive=True))
    matches = 0
    positions = Counter()

    print(f"🔍 Analizujemy potencjalne mecze Pedriego na podstawie events… ({len(files)} plików)")
    for fp in files:
        data = load(fp)
        if not isinstance(data, list):
            continue
        pedri_events = [e for e in data if e.get("player", {}).get("id") == PEDRI_ID]
        if not pedri_events:
            continue
        matches += 1
        pos = None
        for e in data:
            if e.get("type", {}).get("name") == "Starting XI":
                for p in e.get("tactics", {}).get("lineup", []):
                    if p.get("player", {}).get("id") == PEDRI_ID:
                        pos = p.get("position", {}).get("name")
                        break
        if pos:
            positions[pos] += 1

    print(f"✅ Znaleziono {matches} meczów z udziałem Pedriego.")
    if positions:
        print("🧭 Pozycje (liczba meczów):")
        for k, v in positions.most_common():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
