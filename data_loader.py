import json

def load_match_events(filepath):
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)
