from find_pedri_matches import find_pedri_matches_in_events

print("🔍 Wyszukiwanie meczów z Pedrim...")

matches = find_pedri_matches_in_events()

if not matches:
    print("❌ Nie znaleziono żadnych meczów z udziałem Pedriego.")
    exit()

print(f"\nZnaleziono {len(matches)} meczów z udziałem Pedriego.\n")
print("Dostępne mecze Pedriego:")
for i, match in enumerate(matches, 1):
    print(f"{i}. {match['date']} – {match['home_team']} vs {match['away_team']} ({match['file']})")

try:
    choice = int(input("\nWybierz numer meczu do analizy: "))
    selected_match = matches[choice - 1]
    print(f"\n▶️ Wybrano mecz: {selected_match['date']} – {selected_match['home_team']} vs {selected_match['away_team']}")
    print(f"📁 Plik JSON: {selected_match['file']}")
except (ValueError, IndexError):
    print("❌ Wprowadź poprawny numer meczu.")
