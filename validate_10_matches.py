import json
from pathlib import Path

from core.minutes import estimate_minutes

PEDRI_ID = 30486

MATCH_IDS = [
    "3942226",
    "3773466",
    "3773386",
    "3773523",
    "3773457",
    "3773369",
    "3764440",
    "3794686",
    "3795220",
    "3857291",
]

def clock(ev):
    return float(ev.get("minute", 0)) + float(ev.get("second", 0)) / 60

for match_id in MATCH_IDS:
    files = list(Path("data/events").rglob(f"*{match_id}*.json"))

    if not files:
        print(f"{match_id}: FILE NOT FOUND")
        continue

    path = files[0]

    with path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    minutes, details = estimate_minutes(events, PEDRI_ID)

    print("=" * 70)
    print("MATCH:", match_id)
    print("FILE:", path)
    print("CALCULATED MINUTES:", minutes)
    print("DETAILS:", details)

    for ev in events:
        event_type = (ev.get("type") or {}).get("name")

        if event_type == "Substitution":
            player = (ev.get("player") or {}).get("id")
            replacement = ((ev.get("substitution") or {}).get("replacement") or {}).get("id")

            if player == PEDRI_ID or replacement == PEDRI_ID:
                print(
                    "SUB:",
                    round(clock(ev), 3),
                    "OUT:", player,
                    "IN:", replacement,
                )

        if event_type == "Half End":
            print("HALF END:", round(clock(ev), 3))

