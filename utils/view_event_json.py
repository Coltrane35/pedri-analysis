import json

filename = "data/events/3764440.json"  # zmień na inny jeśli chcesz
with open(filename, encoding="utf-8") as f:
    data = json.load(f)

print(f"Typ danych: {type(data)}")
if isinstance(data, list):
    print(f"Liczba wpisów w liście: {len(data)}")
    print("Pierwszy wpis:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))
else:
    print("Dane nie są listą!")
    print(json.dumps(data, indent=2, ensure_ascii=False))
