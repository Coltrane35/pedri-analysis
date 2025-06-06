import os
import json

PEDRI_FULLNAME = "Pedro González López"

EVENTS_DIR = os.path.join("data", "events")
MATCHES_DIR = os.path.join("data", "matches")
OUTPUT_FILE = "pedri_matches.json"

def load_match_metadata():
    metadata = {}
    for subfolder in os.listdir(MATCHES_DIR):
        subdir_path = os.path.join(MATCHES_DIR, subfolder)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(subdir_path, fname)
            with open(fpath, encoding='utf-8') as f:
                matches = json.load(f)
                for match in matches:
                    match_id = match["match_id"]
                    metadata[match_id] = {
                        "date": match.get("match_date"),
                        "home": match.get("home_team", {}).get("home_team_name"),
                        "away": match.get("away_team", {}).get("away_team_name")
                    }
    return metadata

def find_pedri_matches_in_events():
    match_metadata = load_match_metadata()
    found_matches = []

    for fname in os.listdir(EVENTS_DIR):
        if not fname.endswith('.json'):
            continue
        match_id = int(fname.replace(".json", ""))
        fpath = os.path.join(EVENTS_DIR, fname)
        with open(fpath, encoding='utf-8') as f:
            events = json.load(f)
            for event in events:
                try:
                    if "player" in event and event["player"]["name"] == PEDRI_FULLNAME:
                        match_info = match_metadata.get(match_id, {})
                        found_matches.append({
                            "match_id": match_id,
                            "file": fpath,
                            "date": match_info.get("date", "N/A"),
                            "home_team": match_info.get("home", "N/A"),
                            "away_team": match_info.get("away", "N/A")
                        })
                        break
                except (KeyError, TypeError):
                    continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(found_matches, f, indent=2, ensure_ascii=False)

    return found_matches
