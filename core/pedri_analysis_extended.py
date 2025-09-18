# pedri_analysis_extended.py
# Rozszerzona analiza Pedriego:
# - Czyta pedri_profile.json i data/events/<match_id>.json
# - Liczy dodatkowe metryki (final third, penalty area, progressive receptions, deep completions,
#   switches, turnovers, pressures after loss, passes under pressure)
# - Zapisuje CSV per mecz i JSON z podsumowaniem
# - Tworzy wykresy w katalogu plots/ (matplotlib, bez seaborn), z wysoką jakością (PNG + SVG)

import os
import json
import math
import csv
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

EVENTS_DIR = os.path.join("data", "events")
PROFILE_JSON = "pedri_profile.json"
OUT_CSV = "pedri_match_stats_extended.csv"
OUT_JSON = "pedri_summary_extended.json"
PLOTS_DIR = "plots"

PLAYER_ID = 30486
PITCH_X, PITCH_Y = 120.0, 80.0

# Parametry jakości rysunków
DPI = 240
FIGSIZE_TREND = (11, 6)
FIGSIZE_SCATTER = (10, 7)
FIGSIZE_PITCH = (12, 8)
HEAT_BINS = (48, 32)  # gęstsza siatka (zamiast 24x16)
HEAT_INTERP = "bicubic"  # płynne wygładzanie heatmapy


# --------------------
# Pomocnicze funkcje
# --------------------
def load_profile(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        prof = json.load(f)
    out = {}
    for row in prof:
        mid = str(
            row.get("match_id")
            or row.get("id")
            or row.get("matchId")
            or row.get("match")
        )
        if not mid:
            continue
        out[mid] = {
            "team": row.get("team_name") or row.get("team") or "Barcelona/Spain",
            "minutes": float(row.get("minutes") or 90.0),
            "position": row.get("position") or "Unknown",
        }
    return out


def load_events(match_id):
    path = os.path.join(EVENTS_DIR, f"{match_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_completed_pass(ev):
    p = ev.get("pass")
    return bool(p) and ("outcome" not in p)


def distance(a, b):
    if not a or not b:
        return 0.0
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return math.dist((ax, ay), (bx, by))


def toward_goal_delta(start, end, attacking_left_to_right=True):
    if not start or not end:
        return 0.0
    sx, ex = start[0], end[0]
    return (ex - sx) if attacking_left_to_right else (sx - ex)


def infer_attacking_direction(events, team_name):
    """Heurystyka: średnia zmiana x podań celnych w 1. połowie. >=0 → atak w prawo."""
    deltas = []
    for ev in events:
        if ev.get("team", {}).get("name") != team_name:
            continue
        if ev.get("type", {}).get("name") != "Pass":
            continue
        if ev.get("period") != 1:
            continue
        if not is_completed_pass(ev):
            continue
        start = ev.get("location")
        end = (ev.get("pass") or {}).get("end_location")
        if start and end:
            deltas.append(end[0] - start[0])
    if not deltas:
        return True
    return (sum(deltas) / len(deltas)) >= 0.0


def is_progressive_pass(ev, team_attacks_right=True):
    """Progresywne podanie: celne + przybliżenie do bramki >=30% i przesuw x >=10 m."""
    if not is_completed_pass(ev):
        return False
    start = ev.get("location")
    end = (ev.get("pass") or {}).get("end_location")
    if not start or not end:
        return False
    dx = toward_goal_delta(start, end, attacking_left_to_right=team_attacks_right)
    if dx < 10.0:
        return False
    if team_attacks_right:
        dist_start = max(0.0, 120.0 - start[0])
        dist_end = max(0.0, 120.0 - end[0])
    else:
        dist_start = max(0.0, start[0] - 0.0)
        dist_end = max(0.0, end[0] - 0.0)
    gain = dist_start - dist_end
    return (dist_start > 0) and (gain >= 0.3 * dist_start)


def in_final_third(point, team_attacks_right=True):
    """Tercja ataku względem kierunku ataku w danej połowie."""
    if not point:
        return False
    x = point[0]
    return (x >= 80.0) if team_attacks_right else (x <= 40.0)


def in_penalty_area(point, team_attacks_right=True):
    """Pole karne (prostokąt)."""
    if not point:
        return False
    x, y = point[0], point[1]
    if team_attacks_right:
        return (x >= 102.0) and (18.0 <= y <= 62.0)
    else:
        return (x <= 18.0) and (18.0 <= y <= 62.0)


def is_deep_completion(ev, team_attacks_right=True):
    """Celne podanie zakończone <20 m od bramki, nie będące dośrodkowaniem."""
    if not is_completed_pass(ev):
        return False
    pinfo = ev.get("pass") or {}
    if pinfo.get("cross"):
        return False
    end = pinfo.get("end_location")
    if not end:
        return False
    if team_attacks_right:
        dist_to_goal = max(0.0, 120.0 - end[0])
    else:
        dist_to_goal = max(0.0, end[0] - 0.0)
    return dist_to_goal < 20.0


def is_switch_of_play(ev):
    """Zmiana strony: duża zmiana osi Y (>=40 m) + długie podanie (>=30 m) + celne."""
    if not is_completed_pass(ev):
        return False
    start = ev.get("location")
    end = (ev.get("pass") or {}).get("end_location")
    if not start or not end:
        return False
    dy = abs(end[1] - start[1])
    dist = distance(start, end)
    return (dy >= 40.0) and (dist >= 30.0)


def carry_distance(ev):
    c = ev.get("carry")
    if not c:
        return 0.0
    return distance(ev.get("location"), c.get("end_location"))


def is_successful_dribble(ev):
    d = ev.get("dribble")
    return bool(d) and ((d.get("outcome") or {}).get("name") == "Complete")


def per90(total, minutes):
    return (total / minutes * 90.0) if minutes > 0 else 0.0


# --------------------
# Analiza jednego meczu
# --------------------
def analyze_match_extended(match_id, profile_row, directions_cache):
    events = load_events(match_id)
    if not events:
        return None

    pedri_team = profile_row.get("team", "Barcelona/Spain")
    minutes_played = float(profile_row.get("minutes", 90.0))

    # nazwa teamu Pedriego z eventów
    team_candidates = Counter()
    for ev in events:
        if (ev.get("player", {}) or {}).get("id") == PLAYER_ID:
            team_candidates[(ev.get("team", {}) or {}).get("name")] += 1
    team_name = team_candidates.most_common(1)[0][0] if team_candidates else pedri_team

    # kierunek ataku (cache per team)
    if team_name not in directions_cache:
        directions_cache[team_name] = infer_attacking_direction(events, team_name)
    attacks_right_first_half = directions_cache[team_name]

    # cache również dla pozostałych drużyn (do progressive receptions)
    all_teams = {ev.get("team", {}).get("name") for ev in events if ev.get("team")}
    for t in all_teams:
        if t and t not in directions_cache:
            directions_cache[t] = infer_attacking_direction(events, t)

    # indeks podań po id (do xA)
    pass_by_id = {
        ev.get("id"): ev for ev in events if ev.get("type", {}).get("name") == "Pass"
    }

    # liczniki
    cnt = defaultdict(int)
    acc = defaultdict(float)

    # do „pressures after loss”
    pedri_losses_times = []  # (period, minute_with_seconds)

    # dane do heatmap
    pass_starts = []
    carry_starts = []

    # 1) pętla po eventach — akcje Pedriego i xA po key_pass_id
    for ev in events:
        etype = (ev.get("type") or {}).get("name")

        # xA (key_pass_id) — strzały drużyny
        if etype == "Shot":
            xg = float((ev.get("shot") or {}).get("statsbomb_xg") or 0.0)
            acc["team_total_xg"] += xg
            kpid = (ev.get("shot") or {}).get("key_pass_id")
            if kpid and kpid in pass_by_id:
                p = pass_by_id[kpid]
                if (p.get("player", {}) or {}).get("id") == PLAYER_ID:
                    acc["xa"] += xg

        # odtąd tylko eventy Pedriego
        if (ev.get("player", {}) or {}).get("id") != PLAYER_ID:
            continue

        period = ev.get("period")
        tstamp = float(ev.get("minute", 0)) + float(ev.get("second", 0)) / 60.0
        team_attacks_right = (
            attacks_right_first_half if period == 1 else (not attacks_right_first_half)
        )

        if etype in ("Miscontrol", "Dispossessed"):
            cnt["turnovers"] += 1
            pedri_losses_times.append((period, tstamp))

        if etype == "Pass":
            cnt["passes_total"] += 1
            if is_completed_pass(ev):
                cnt["passes_completed"] += 1
                if ev.get("location"):
                    pass_starts.append(tuple(ev["location"]))

                pinfo = ev.get("pass") or {}
                end = pinfo.get("end_location")

                if (
                    pinfo.get("assisted_shot_id")
                    or pinfo.get("shot_assist")
                    or pinfo.get("goal_assist")
                ):
                    cnt["key_passes"] += 1

                if ev.get("under_pressure"):
                    cnt["passes_under_pressure"] += 1

                if end:
                    if in_final_third(end, team_attacks_right):
                        cnt["passes_into_final_third"] += 1
                    if in_penalty_area(end, team_attacks_right):
                        cnt["passes_into_penalty_area"] += 1

                if is_deep_completion(ev, team_attacks_right):
                    cnt["deep_completions"] += 1

                if is_progressive_pass(ev, team_attacks_right=team_attacks_right):
                    cnt["progressive_passes"] += 1

                if is_switch_of_play(ev):
                    cnt["switches_of_play"] += 1

        elif etype == "Shot":
            cnt["shots"] += 1
            if (ev.get("shot") or {}).get("outcome", {}).get("name") == "Goal":
                cnt["goals"] += 1
            acc["xg"] += float((ev.get("shot") or {}).get("statsbomb_xg") or 0.0)
            if (ev.get("shot") or {}).get("key_pass_id"):
                cnt["shots_after_key_pass"] += 1

        elif etype == "Dribble":
            cnt["dribbles_attempted"] += 1
            if is_successful_dribble(ev):
                cnt["dribbles_completed"] += 1

        elif etype == "Carry":
            if ev.get("location"):
                carry_starts.append(tuple(ev["location"]))
            acc["carry_distance"] += carry_distance(ev)
            start = ev.get("location")
            end = (ev.get("carry") or {}).get("end_location")
            if start and end:
                dx = toward_goal_delta(
                    start, end, attacking_left_to_right=team_attacks_right
                )
                if dx >= 10.0:
                    cnt["progressive_carries"] += 1

        elif etype == "Pressure":
            cnt["pressures"] += 1
        elif etype == "Tackle":
            cnt["tackles"] += 1
        elif etype == "Interception":
            cnt["interceptions"] += 1
        elif etype == "Ball Recovery":
            cnt["ball_recoveries"] += 1

    # 2) progressive receptions — z P O D A Ń do Pedriego
    for ev in events:
        if ev.get("type", {}).get("name") != "Pass":
            continue
        pinfo = ev.get("pass") or {}
        rec = pinfo.get("recipient") or {}
        if rec.get("id") != PLAYER_ID:
            continue
        team_name_pass = (ev.get("team") or {}).get("name")
        if not team_name_pass:
            continue
        team_attacks_right_src = directions_cache.get(team_name_pass, True)
        if is_completed_pass(ev) and is_progressive_pass(
            ev, team_attacks_right=team_attacks_right_src
        ):
            cnt["progressive_receptions"] += 1

    # 3) pressures after loss — pres w 5s po własnej stracie
    losses_sorted = sorted(pedri_losses_times)
    loss_idx = 0
    for ev in events:
        if (ev.get("player", {}) or {}).get("id") != PLAYER_ID:
            continue
        if (ev.get("type") or {}).get("name") != "Pressure":
            continue
        period = ev.get("period")
        tstamp = float(ev.get("minute", 0)) + float(ev.get("second", 0)) / 60.0

        while loss_idx < len(losses_sorted) and (
            (losses_sorted[loss_idx][0] < period)
            or (
                losses_sorted[loss_idx][0] == period
                and losses_sorted[loss_idx][1] < tstamp - (5.0 / 60.0)
            )
        ):
            loss_idx += 1

        candidates = []
        if loss_idx < len(losses_sorted):
            candidates.append(losses_sorted[loss_idx])
        if loss_idx - 1 >= 0:
            candidates.append(losses_sorted[loss_idx - 1])

        for p_per, p_t in candidates:
            if p_per == period and 0.0 <= (tstamp - p_t) <= (5.0 / 60.0):
                cnt["pressures_after_loss"] += 1
                break

    # procenty
    passes_total = cnt["passes_total"]
    passes_completed = cnt["passes_completed"]
    pass_pct = (passes_completed / passes_total * 100.0) if passes_total else 0.0

    drib_att = cnt["dribbles_attempted"]
    drib_succ = cnt["dribbles_completed"]
    drib_pct = (drib_succ / drib_att * 100.0) if drib_att else 0.0

    row = {
        "match_id": match_id,
        "team": team_name,
        "minutes": round(minutes_played, 1),
        # bazowe
        "passes_total": passes_total,
        "passes_completed": passes_completed,
        "pass_pct": round(pass_pct, 1),
        "key_passes": cnt["key_passes"],
        "shots": cnt["shots"],
        "goals": cnt["goals"],
        "xg": round(acc["xg"], 3),
        "xa": round(acc["xa"], 3),
        "shots_after_key_pass": cnt["shots_after_key_pass"],
        "dribbles_attempted": drib_att,
        "dribbles_completed": drib_succ,
        "dribbles_pct": round(drib_pct, 1),
        "pressures": cnt["pressures"],
        "tackles": cnt["tackles"],
        "interceptions": cnt["interceptions"],
        "ball_recoveries": cnt["ball_recoveries"],
        "progressive_passes": cnt["progressive_passes"],
        "progressive_carries": cnt["progressive_carries"],
        "carry_distance": round(acc["carry_distance"], 1),
        "team_total_xg": round(acc["team_total_xg"], 3),
        # rozszerzone
        "passes_under_pressure": cnt["passes_under_pressure"],
        "passes_into_final_third": cnt["passes_into_final_third"],
        "passes_into_penalty_area": cnt["passes_into_penalty_area"],
        "progressive_receptions": cnt["progressive_receptions"],
        "deep_completions": cnt["deep_completions"],
        "switches_of_play": cnt["switches_of_play"],
        "turnovers": cnt["turnovers"],
        "pressures_after_loss": cnt["pressures_after_loss"],
    }

    heat = {
        "pass_starts": pass_starts,
        "carry_starts": carry_starts,
    }
    return row, heat


# --------------------
# Rysowanie (wysoka jakość)
# --------------------
def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _save_both(fig, out_png, out_svg):
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def _draw_pitch(ax):
    """Rysuje linie boiska 120x80."""
    ax.add_patch(
        Rectangle((0, 0), 120, 80, fill=False, linewidth=1.5, antialiased=True)
    )
    ax.plot([60, 60], [0, 80], linewidth=1.2)
    center = Circle((60, 40), 9.15, fill=False, linewidth=1.2)
    ax.add_patch(center)
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, linewidth=1.2))
    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, linewidth=1.2))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False, linewidth=1.0))
    ax.plot(108, 40, marker="o", markersize=2)
    ax.plot(12, 40, marker="o", markersize=2)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Długość boiska (x)")
    ax.set_ylabel("Szerokość boiska (y)")


def plot_trends(match_rows):
    """Trend KP/90 i PP/90 — większe figury, grubsze linie, siatka."""
    ensure_dir(PLOTS_DIR)
    mins = [r["minutes"] for r in match_rows]
    kp90 = [
        (r["key_passes"] / m * 90.0) if m > 0 else 0.0 for r, m in zip(match_rows, mins)
    ]
    pp90 = [
        (r["progressive_passes"] / m * 90.0) if m > 0 else 0.0
        for r, m in zip(match_rows, mins)
    ]
    x = np.arange(1, len(match_rows) + 1)

    fig = plt.figure(figsize=FIGSIZE_TREND)
    ax = fig.add_subplot(111)
    ax.plot(
        x,
        kp90,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="Key Passes/90",
        antialiased=True,
    )
    ax.plot(
        x,
        pp90,
        marker="s",
        linewidth=2.2,
        markersize=5,
        label="Progressive Passes/90",
        antialiased=True,
    )
    ax.set_xlabel("Mecz (indeks)")
    ax.set_ylabel("Wartość per 90")
    ax.set_title("Trendy: KP/90 i Progressive Passes/90")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    _save_both(
        fig,
        os.path.join(PLOTS_DIR, "trend_kp_pp_per90.png"),
        os.path.join(PLOTS_DIR, "trend_kp_pp_per90.svg"),
    )


def plot_scatter_pp_vs_pc(match_rows):
    """Scatter: PP vs PC — większe DPI, półprzezroczyste markery, obrysy, siatka."""
    ensure_dir(PLOTS_DIR)
    pp = [r["progressive_passes"] for r in match_rows]
    pc = [r["progressive_carries"] for r in match_rows]
    xa = [max(20.0, r["xa"] * 300.0) for r in match_rows]  # większa baza rozmiaru

    fig = plt.figure(figsize=FIGSIZE_SCATTER)
    ax = fig.add_subplot(111)
    ax.scatter(
        pp, pc, s=xa, alpha=0.55, linewidths=0.8, edgecolors="face", antialiased=True
    )
    ax.set_xlabel("Progressive Passes (per mecz)")
    ax.set_ylabel("Progressive Carries (per mecz)")
    ax.set_title("PP vs PC (rozmiar punktu ~ xA)")
    ax.grid(True, linestyle="--", alpha=0.35)
    _save_both(
        fig,
        os.path.join(PLOTS_DIR, "scatter_pp_vs_pc.png"),
        os.path.join(PLOTS_DIR, "scatter_pp_vs_pc.svg"),
    )


def plot_heatmap_points(points, title, out_name, bins=HEAT_BINS):
    """Heatmapa 2D na boisku 120x80 z wygładzaniem i pitch overlay."""
    ensure_dir(PLOTS_DIR)
    fig = plt.figure(figsize=FIGSIZE_PITCH)
    ax = fig.add_subplot(111)

    if not points:
        _draw_pitch(ax)
        ax.set_title(f"{title}\n(Brak danych)")
        _save_both(
            fig,
            os.path.join(PLOTS_DIR, out_name),
            os.path.join(PLOTS_DIR, out_name.replace(".png", ".svg")),
        )
        return

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    H, xedges, yedges = np.histogram2d(
        xs, ys, bins=bins, range=[[0, PITCH_X], [0, PITCH_Y]]
    )
    H = np.log1p(H)  # kompresja zakresu, lepsza widoczność słabszych obszarów

    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[0, PITCH_X, 0, PITCH_Y],
        aspect="equal",
        interpolation=HEAT_INTERP,
    )

    _draw_pitch(ax)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Natężenie (log1p)")

    _save_both(
        fig,
        os.path.join(PLOTS_DIR, out_name),
        os.path.join(PLOTS_DIR, out_name.replace(".png", ".svg")),
    )


# --------------------
# Main
# --------------------
def main():
    if not os.path.exists(PROFILE_JSON):
        print(f"⚠️ Brak pliku {PROFILE_JSON}. Uruchom najpierw pedri_profile.py")
        return

    profile = load_profile(PROFILE_JSON)
    match_ids = sorted(profile.keys(), key=lambda k: int(k))
    print("🔍 Rozszerzona analiza — wczytuję mecze z pedri_profile.json")
    print(f"Znaleziono {len(match_ids)} meczów.\n")

    rows = []
    total_minutes = 0.0
    agg = defaultdict(float)

    all_pass_starts = []
    all_carry_starts = []

    directions_cache = {}

    for i, mid in enumerate(match_ids, 1):
        info = profile[mid]
        print(f"{i}. Analiza meczu {mid} ...")
        res = analyze_match_extended(mid, info, directions_cache)
        if not res:
            print(f"   ⚠️ Brak eventów dla {mid}")
            continue
        row, heat = res
        rows.append(row)
        total_minutes += float(row["minutes"])

        for k, v in row.items():
            if k in {"match_id", "team", "minutes"}:
                continue
            if isinstance(v, (int, float)):
                agg[k] += float(v)

        all_pass_starts.extend(heat["pass_starts"])
        all_carry_starts.extend(heat["carry_starts"])

    if not rows:
        print("⚠️ Brak danych do zapisania.")
        return

    # zapis CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # podsumowanie (totals + per90)
    summary = {
        "matches": len(rows),
        "minutes": round(total_minutes, 1),
        "totals": {},
        "per90": {},
    }
    for k, v in agg.items():
        summary["totals"][k] = round(v, 3)
        summary["per90"][k] = round(per90(v, total_minutes), 3)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # wykresy wysokiej jakości
    plot_trends(rows)
    plot_scatter_pp_vs_pc(rows)
    plot_heatmap_points(
        all_pass_starts, "Heatmapa startów podań (Pedri)", "heatmap_pass_starts.png"
    )
    plot_heatmap_points(
        all_carry_starts, "Heatmapa startów carry (Pedri)", "heatmap_carry_starts.png"
    )

    # skrót
    passes_total = int(summary["totals"].get("passes_total", 0))
    passes_completed = int(summary["totals"].get("passes_completed", 0))
    pass_pct = (passes_completed / passes_total * 100.0) if passes_total else 0.0
    print("\n📊 PODSUMOWANIE (extended):")
    print(f"   Minuty: {summary['minutes']}")
    print(f"   Podania: {passes_completed}/{passes_total} ({pass_pct:.1f}%)")
    print(
        f"   KP: {int(summary['totals'].get('key_passes',0))} | xA: {summary['totals'].get('xa',0.0):.3f}"
    )
    print(
        f"   PP: {int(summary['totals'].get('progressive_passes',0))} | PC: {int(summary['totals'].get('progressive_carries',0))}"
    )
    print(
        f"   Final third: {int(summary['totals'].get('passes_into_final_third',0))} | Pen area: {int(summary['totals'].get('passes_into_penalty_area',0))}"
    )
    print(
        f"   Deep completions: {int(summary['totals'].get('deep_completions',0))} | Switches: {int(summary['totals'].get('switches_of_play',0))}"
    )
    print(
        f"   Progressive receptions: {int(summary['totals'].get('progressive_receptions',0))}"
    )
    print(
        f"   Turnovers: {int(summary['totals'].get('turnovers',0))} | Pressures after loss: {int(summary['totals'].get('pressures_after_loss',0))}"
    )
    print(f"\n💾 Zapisano: {OUT_CSV}, {OUT_JSON} oraz wykresy w {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
