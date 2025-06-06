# Metodologia analizy: Pierwiastek DNA Barçy w grze Pedriego
        
##  Cel projektu

Celem niniejszego projektu jest przeanalizowanie stylu gry Pedro González Lópeza (Pedriego) w kontekście tzw. **pierwiastka DNA Barçy**, inspirowanego filozofią piłkarską FC Barcelony ukształtowaną przez Johana Cruyffa, rozwijaną przez Pepa Guardiolę i obecną w szkoleniu La Masii oraz w ideach Xaviego Hernándeza.

##  Czym jest "DNA Barçy"?

DNA Barçy to nieformalna, ale jasno określona filozofia gry, której kluczowe cechy to:
        - **Percepcja**  szybka orientacja przestrzenna, skanowanie otoczenia, zrozumienie sytuacji.
        - **Decyzja**  odpowiedni wybór zagrania, unikanie ryzyka przy zachowaniu progresji.
        - **Wykonanie**  techniczna perfekcja, timing, umiejętność utrzymania się przy piłce.
        
Źródła inspiracji:
        - [Barça Innovation Hub](https://barcainnovationhub.com)
        - La Masia training methodology
        - Wypowiedzi Guardioli, Xaviego, Roury i Valverde nt. filozofii gry.
        
##  Zakres analizy

Analizujemy 49 meczów Pedriego dostępnych w bazie StatsBomb (`data/events`), z sezonów od debiutu w FC Barcelonie do 2023. W projekcie użyto danych eventowych (pass, pressure, carry, shot itd.).

Analiza koncentruje się na:
        - **Pozycjonowaniu** (heatmapy, progressive actions),
        - **Zagraniach pod presją**,
        - **Sekwencjach podań (np. "third-man runs")**,
        - **Reakcji na stratę piłki (counter-pressing)**,
        - **Zachowaniach bez piłki (movement off the ball)**.
        
##  Praktyczne zastosowanie

Wnioski mogą posłużyć jako:
        - Porównanie z legendami La Masii (Xavi, Iniesta),
        - Model selekcji zawodników pasujących do DNA Barçy,
        - Baza do stworzenia narzędzi scoutingowych (np. scoring modelu gracza).
        
##  Możliwe rozszerzenia

- Dodanie meczów z `data/matches` (po identyfikacji eventów),
- Scraping danych z Transfermarkt / FBref (minuty, kontuzje),
- Porównanie Pedriego z innymi zawodnikami spoza La Masii (np. Kimmich, Bellingham),
- Dashboard Streamlit z interaktywną analizą.

##  Stan projektu

Dane: 49 meczów Pedriego z katalogu `data/events`.

Repozytorium: [link do GitHuba]

Autor: Grzegorz Dariusz Rączka