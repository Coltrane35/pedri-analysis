# core/pedri_analysis.py
# -*- coding: utf-8 -*-
"""
Pedri analysis over StatsBomb-like events.

- Scans data/events/*.json (and subfolders)
- Extracts Pedri (player_id=30486) events
- Computes per-match + aggregated stats
- Saves CSV + charts into outputs/csv and outputs/figures

Conventions (canonical outputs):
- outputs/csv/pedri_match_stats.csv               -> basic per-match columns
- outputs/csv/pedri_match_stats_extended.csv      -> full per-match columns (incl. per90)
- outputs/csv/pedri_summary.csv                   -> one-row aggregate
- (compat) outputs/csv/pedri_per_match_stats.csv  -> alias to extended (can be removed later)

Viz polish:
- Bar charts with value labels, Top-20 clipping, PNG + SVG
- Hexbin heatmap over a simple pitch
- Radar charts (RAW p90 + percentiles)
- Pass maps (all passes + progressive only)
- Robust match_id from filename (digits in stem), plus source_file for traceability
"""

import json, os, glob, re, math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# ----------------- config -----------------
PEDRI_ID   = 30486
EVENTS_DIR = os.path.join("data", "events")

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
CSV_DIR = os.path.join(OUT_DIR, "csv")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

FIG_SIZE = (10, 6)
CMAP = "viridis"          # colormap for heatmap
TOP_N_DEFAULT = 20        # bar charts show Top-20 by default

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "figure.figsize": (9, 5.5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ----------------- helpers -----------------
def safe_get(d, path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def is_pedri_event(ev):   return safe_get(ev, ("player", "id")) == PEDRI_ID
def is_pass(ev):          return safe_get(ev, ("type", "name")) == "Pass"
def is_completed_pass(ev):return is_pass(ev) and safe_get(ev, ("pass", "outcome", "name")) in (None, "Complete")
def is_shot(ev):          return safe_get(ev, ("type", "name")) == "Shot"
def is_dribble(ev):       return safe_get(ev, ("type", "name")) == "Dribble"
def dribble_outcome(ev):  return safe_get(ev, ("dribble", "outcome", "name"))
def is_carry(ev):         return safe_get(ev, ("type", "name")) == "Carry"
def is_pressure(ev):      return safe_get(ev, ("type", "name")) == "Pressure"
def is_tackle(ev):        return safe_get(ev, ("type", "name")) == "Duel" and safe_get(ev, ("duel", "type", "name")) == "Tackle"
def is_interception(ev):  return safe_get(ev, ("type", "name")) == "Interception"
def is_ball_recovery(ev): return safe_get(ev, ("type", "name")) == "Ball Recovery"

def is_key_pass(ev):
    if not is_pass(ev): return False
    return safe_get(ev, ("pass", "shot_assist")) is True or safe_get(ev, ("pass", "assisted_shot_id")) is not None

def get_loc(ev):
    loc = ev.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None

def get_end_loc(ev):
    end = ev.get("pass", {}).get("end_location") if is_pass(ev) else ev.get("carry", {}).get("end_location") if is_carry(ev) else None
    if isinstance(end, list) and len(end) >= 2:
        return float(end[0]), float(end[1])
    return None

def is_progressive_pass(ev):
    if not is_pass(ev): return False
    start, end = get_loc(ev), get_end_loc(ev)
    if not start or not end: return False
    dx = end[0] - start[0]
    dy = abs(end[1] - start[1])
    return dx >= 10 and dx > dy

def carry_distance(ev):
    if not is_carry(ev): return 0.0
    start, end = get_loc(ev), get_end_loc(ev)
    if not start or not end: return 0.0
    return math.dist(start, end)

def event_minute(ev):
    m, s = ev.get("minute", 0), ev.get("second", 0)
    try: return int(m) + int(s)/60.0
    except: return 0.0

def match_date_from_events(events):
    for ev in events:
        dt = safe_get(ev, ("match_date",)) or safe_get(ev, ("match", "date"))
        if dt: return dt
    return None

def lineup_position_for_pedri(events):
    for ev in events:
        if safe_get(ev, ("type", "name")) == "Starting XI":
            for p in safe_get(ev, ("tactics", "lineup"), []) or []:
                if safe_get(p, ("player", "id")) == PEDRI_ID:
                    pos = safe_get(p, ("position", "name"))
                    if pos: return pos
    return None

def minutes_played_estimate(events):
    on_min, off_min = 0.0, None
    for ev in events:
        if safe_get(ev, ("type", "name")) == "Substitution":
            out_player = safe_get(ev, ("substitution", "replacement", "id"))
            in_player  = safe_get(ev, ("player", "id"))
            minute     = event_minute(ev)
            if in_player == PEDRI_ID:  off_min = minute
            if out_player == PEDRI_ID: on_min = minute
    if off_min is not None:
        return max(0.0, min(95.0, off_min - on_min))
    times = [event_minute(e) for e in events if is_pedri_event(e)]
    if times:
        est = max(times) - min(times)
        return max(10.0, min(95.0, est))
    return 0.0

def match_id_from_filename(file_hint: str | None) -> str:
    """Robust: pull digits from filename stem; fallback to stem."""
    if not file_hint: return "unknown"
    stem = Path(file_hint).stem
    m = re.search(r"\d+", stem)
    return m.group(0) if m else stem

# ----------------- core compute -----------------
def compute_stats_for_match(events, file_hint: str | None = None):
    pedri_events = [e for e in events if is_pedri_event(e)]
    if not pedri_events:
        return None

    match_id   = match_id_from_filename(file_hint)  # always from filename
    match_date = match_date_from_events(events)
    position   = lineup_position_for_pedri(events)
    minutes    = minutes_played_estimate(events)

    total_pass = sum(is_pass(e) for e in pedri_events)
    comp_pass  = sum(is_completed_pass(e) for e in pedri_events)
    key_passes = sum(is_key_pass(e) for e in pedri_events)
    prog_pass  = sum(is_progressive_pass(e) for e in pedri_events)

    shots = sum(is_shot(e) for e in pedri_events)
    xg    = sum(safe_get(e, ("shot", "statsbomb_xg"), 0.0) or 0.0 for e in pedri_events if is_shot(e))

    drib_att  = sum(is_dribble(e) for e in pedri_events)
    drib_succ = sum(1 for e in pedri_events if is_dribble(e) and dribble_outcome(e) == "Complete")

    carries   = sum(is_carry(e) for e in pedri_events)
    carry_m   = sum(carry_distance(e) for e in pedri_events)

    pressures = sum(is_pressure(e) for e in pedri_events)
    tackles   = sum(is_tackle(e) for e in pedri_events)
    intercp   = sum(is_interception(e) for e in pedri_events)
    recover   = sum(is_ball_recovery(e) for e in pedri_events)

    pass_pct  = (comp_pass / total_pass * 100.0) if total_pass else 0.0
    per90     = (lambda x: (x / minutes * 90.0) if minutes > 0 else 0.0)

    return {
        "match_id": match_id,
        "match_date": match_date,
        "position": position,
        "minutes": round(minutes, 1),

        "passes_attempted": int(total_pass),
        "passes_completed": int(comp_pass),
        "pass_pct": round(pass_pct, 1),
        "key_passes": int(key_passes),
        "progressive_passes": int(prog_pass),

        "shots": int(shots),
        "xg": round(xg, 3),

        "dribbles_attempted": int(drib_att),
        "dribbles_completed": int(drib_succ),

        "carries": int(carries),
        "carry_distance_units": round(carry_m, 1),

        "pressures": int(pressures),
        "tackles": int(tackles),
        "interceptions": int(intercp),
        "ball_recoveries": int(recover),

        "key_passes_p90": round(per90(key_passes), 2),
        "prog_passes_p90": round(per90(prog_pass), 2),
        "dribbles_completed_p90": round(per90(drib_succ), 2),
        "pressures_p90": round(per90(pressures), 2),
        "tackles_p90": round(per90(tackles), 2),
        "interceptions_p90": round(per90(intercp), 2),
        "ball_recoveries_p90": round(per90(recover), 2),
        "shots_p90": round(per90(shots), 2),
        "xg_p90": round(per90(xg), 3),

        "source_file": str(file_hint) if file_hint else None,
    }

# ----------------- IO -----------------
def load_events_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data
    except Exception as e:
        print(f"⚠️  Could not parse {path}: {e}")
    return []

def collect_all_events():
    files = sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json")))
    files += sorted(glob.glob(os.path.join(EVENTS_DIR, "**", "*.json"), recursive=True))
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            uniq.append(f); seen.add(f)
    return uniq

# ----------------- viz helpers -----------------
def _savefig(fig, basename: str):
    """Save both PNG and SVG for crisp docs/PR into outputs/figures."""
    png = os.path.join(FIG_DIR, basename)
    svg = os.path.join(FIG_DIR, Path(basename).with_suffix(".svg").name)
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved: {png} + {svg}")

def _annotate_bars(ax):
    for p in ax.patches:
        try:
            val = p.get_height()
            if pd.isna(val):
                continue
            ax.annotate(
                f"{val:.0f}" if abs(val) >= 1 else f"{val:.2f}",
                (p.get_x() + p.get_width() / 2, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9
            )
        except Exception:
            continue

def plot_bar(series: pd.Series, title: str, xlabel: str, outname: str, top_n: int = TOP_N_DEFAULT):
    s = series.dropna().copy()
    if len(s) == 0:
        print(f"⚠️  No data for {title}")
        return
    s = s.sort_values(ascending=False)
    if len(s) > top_n:
        s = s.iloc[:top_n]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    s.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)

    # ✅ Matplotlib-compatible: obrót w tick_params, wyrównanie osobno
    ax.tick_params(axis="x", labelrotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

    _annotate_bars(ax)
    plt.tight_layout()
    _savefig(fig, outname)


def plot_hist(values, title, xlabel, outname, bins="auto"):
    vals = pd.Series(values).dropna().values
    if vals.size == 0:
        print(f"⚠️  No data for {title}")
        return
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.hist(vals, bins=bins, edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
    plt.tight_layout()
    _savefig(fig, outname)

def _draw_pitch(ax):
    """Minimalist pitch 120x80 (StatsBomb-like)."""
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, linewidth=1.2))
    ax.plot([60, 60], [0, 80], linewidth=1.0)
    ax.add_patch(Circle((60, 40), 9.15, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False, linewidth=0.9))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, linewidth=0.9))
    ax.plot(11, 40, marker="o", markersize=2)
    ax.plot(109, 40, marker="o", markersize=2)
    ax.set_xlim(0, 120); ax.set_ylim(0, 80); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Pitch X (0→120)"); ax.set_ylabel("Pitch Y (0→80)")

def plot_heatmap_locations(events, title, outname, gridsize=30, mincnt=1):
    locs = np.array([get_loc(e) for e in events if is_pedri_event(e) and get_loc(e) is not None], dtype=float)
    if locs.size == 0:
        print("⚠️  No locations for heatmap."); return
    x, y = locs[:, 0], locs[:, 1]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _draw_pitch(ax)
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=CMAP, mincnt=mincnt, alpha=0.9)
    cb = fig.colorbar(hb, ax=ax); cb.set_label("Event density")
    ax.set_title(title); ax.grid(False)
    plt.tight_layout()
    _savefig(fig, outname)

# === RADAR: konfiguracja i helpery ===
METRICS_RADAR = [
    "key_passes_p90",
    "prog_passes_p90",
    "dribbles_completed_p90",
    "pressures_p90",
    "tackles_p90",
    "interceptions_p90",
    "shots_p90",
    "xg_p90",
]
RADAR_LABELS = {
    "key_passes_p90": "KeyP p90",
    "prog_passes_p90": "ProgP p90",
    "dribbles_completed_p90": "Dribbles p90",
    "pressures_p90": "Pressures p90",
    "tackles_p90": "Tackles p90",
    "interceptions_p90": "Interceptions p90",
    "shots_p90": "Shots p90",
    "xg_p90": "xG p90",
}

def _radar_polar_axes(n_axes: int, rmax=None, rlabel=""):
    import numpy as _np, math as _math, matplotlib.pyplot as _plt
    angles = _np.linspace(0, 2 * _math.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close
    fig = _plt.figure(figsize=(7.5, 7.5))
    ax = _plt.subplot(111, polar=True)
    ax.set_theta_offset(_math.pi / 2)
    ax.set_theta_direction(-1)
    if rmax is not None:
        ax.set_rlim(0, rmax)
    ax.set_rlabel_position(0)
    ax.grid(True, linestyle=":", linewidth=0.7)
    if rlabel:
        ax.set_ylabel(rlabel)
    return fig, ax, angles

def _radar_plot(values, labels, title, outname, rmax=None, rlabel=""):
    assert len(values) == len(labels)
    fig, ax, angles = _radar_polar_axes(len(labels), rmax=rmax, rlabel=rlabel)
    vals = list(values) + [values[0]]
    ax.set_thetagrids(np.degrees(np.array(angles[:-1])), labels)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.15)
    ax.set_title(title, pad=20)
    _savefig(fig, outname)

def _percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna().values
    if s.size == 0 or pd.isna(value):
        return float("nan")
    return float((s <= value).mean() * 100.0)

def generate_pedri_radars(df: pd.DataFrame):
    """Draw two radars from per-match df (RAW p90 + percentiles)."""
    cols = [c for c in METRICS_RADAR if c in df.columns]
    if not cols:
        print("⚠️  Radar: no p90 columns – skipping.")
        return
    raw_mean = df[cols].mean(numeric_only=True)
    raw_vals = raw_mean.values.astype(float)
    axis_labels = [RADAR_LABELS.get(c, c) for c in cols]
    rmax = max(1.0, float(np.nanmax(raw_vals)) * 1.1)
    _radar_plot(raw_vals, axis_labels, "Pedri – RAW per 90 (mean)", "pedri_radar_p90_raw.png", rmax=rmax, rlabel="p90")

    pct_vals = [_percentile_rank(df[c], raw_mean[c]) for c in cols]
    _radar_plot(np.array(pct_vals, dtype=float), axis_labels,
                "Pedri – Percentile per 90 (vs own matches)",
                "pedri_radar_p90_percentile.png", rmax=100.0, rlabel="percentile")

# === PASS MAPS ===
def _collect_pedri_pass_segments(events):
    """Return arrays of starts/ends for Pedri's passes."""
    xs, ys, xe, ye, prog_flags = [], [], [], [], []
    for e in events:
        if not (is_pedri_event(e) and is_pass(e)):
            continue
        start = get_loc(e)
        end = get_end_loc(e)
        if not start or not end:
            continue
        xs.append(start[0]); ys.append(start[1])
        xe.append(end[0]);   ye.append(end[1])
        prog_flags.append(is_progressive_pass(e))
    return np.array(xs), np.array(ys), np.array(xe), np.array(ye), np.array(prog_flags, dtype=bool)

def _plot_pass_map(xs, ys, xe, ye, title, outname, alpha=0.6, width=0.002):
    """Draw arrows (quiver) from (x,y) to (x2,y2)."""
    if xs.size == 0:
        print(f"⚠️  Pass map: no passes for {title}")
        return
    dx = xe - xs
    dy = ye - ys
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _draw_pitch(ax)
    ax.quiver(xs, ys, dx, dy, angles='xy', scale_units='xy', scale=1, width=width, alpha=alpha)
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()
    _savefig(fig, outname)

def generate_pass_maps(pedri_events_all):
    xs, ys, xe, ye, prog = _collect_pedri_pass_segments(pedri_events_all)
    if xs.size == 0:
        print("⚠️  No Pedri passes found – skipping pass maps.")
        return
    # All passes
    _plot_pass_map(xs, ys, xe, ye, "Pedri – Pass Map (all passes)", "pedri_pass_map_all.png", alpha=0.5)
    # Progressive-only
    mask = prog
    _plot_pass_map(xs[mask], ys[mask], xe[mask], ye[mask], "Pedri – Pass Map (progressive only)", "pedri_pass_map_progressive.png", alpha=0.8)

# ----------------- main -----------------
def main():
    print("🔍 Scanning events for Pedri…")
    files = collect_all_events()
    if not files:
        print("❌ No event files found in data/events/"); return

    per_match, all_pedri_events = [], []
    for fp in files:
        events = load_events_file(fp)
        if not events: continue
        stats = compute_stats_for_match(events, file_hint=fp)
        if stats:
            per_match.append(stats)
            all_pedri_events.extend([e for e in events if is_pedri_event(e)])

    if not per_match:
        print("❌ No Pedri events found across files."); return

    df = pd.DataFrame(per_match)
    df["match_id"] = df["match_id"].astype(str)

    def parse_dt(x):
        if x:
            for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y"):
                try: return datetime.strptime(x, fmt)
                except: continue
        return None
    if "match_date" in df.columns:
        df["_sort_dt"] = df["match_date"].apply(parse_dt)
        df = df.sort_values(by=["_sort_dt", "match_id"], ascending=True).drop(columns=["_sort_dt"], errors="ignore")

    # ---- CSV exports ----
    basic_cols = [
        "match_id","match_date","position","minutes",
        "passes_attempted","passes_completed","pass_pct",
        "key_passes","progressive_passes","shots","xg"
    ]
    basic = df[[c for c in basic_cols if c in df.columns]]
    basic_path = os.path.join(CSV_DIR, "pedri_match_stats.csv")
    basic.to_csv(basic_path, index=False, encoding="utf-8"); print(f"💾 Saved: {basic_path}")

    extended_path = os.path.join(CSV_DIR, "pedri_match_stats_extended.csv")
    df.to_csv(extended_path, index=False, encoding="utf-8"); print(f"💾 Saved: {extended_path}")

    compat_path = os.path.join(CSV_DIR, "pedri_per_match_stats.csv")
    df.to_csv(compat_path, index=False, encoding="utf-8"); print(f"💾 Saved (compat): {compat_path}")

    # ---- Summary ----
    agg = {
        "matches": len(df),
        "minutes_total": float(df["minutes"].sum()),
        "passes_attempted": int(df["passes_attempted"].sum()),
        "passes_completed": int(df["passes_completed"].sum()),
        "pass_pct_weighted": round(100.0 * df["passes_completed"].sum() / df["passes_attempted"].sum(), 2) if df["passes_attempted"].sum() else 0.0,
        "key_passes": int(df["key_passes"].sum()),
        "progressive_passes": int(df["progressive_passes"].sum()),
        "shots": int(df["shots"].sum()),
        "xg": round(float(df["xg"].sum()), 3),
        "dribbles_attempted": int(df["dribbles_attempted"].sum()),
        "dribbles_completed": int(df["dribbles_completed"].sum()),
        "carries": int(df["carries"].sum()),
        "carry_distance_units": round(float(df["carry_distance_units"].sum()), 1),
        "pressures": int(df["pressures"].sum()),
        "tackles": int(df["tackles"].sum()),
        "interceptions": int(df["interceptions"].sum()),
        "ball_recoveries": int(df["ball_recoveries"].sum()),
    }
    summary_path = os.path.join(CSV_DIR, "pedri_summary.csv")
    pd.DataFrame([agg]).to_csv(summary_path, index=False, encoding="utf-8"); print(f"💾 Saved: {summary_path}")

    # ---- Charts (Top-20 where relevant) ----
    plot_bar(df.set_index("match_id")["key_passes"],         "Key passes per match (Top 20)",         "Match ID", "pedri_key_passes_per_match.png", top_n=TOP_N_DEFAULT)
    plot_bar(df.set_index("match_id")["progressive_passes"], "Progressive passes per match (Top 20)", "Match ID", "pedri_prog_passes_per_match.png", top_n=TOP_N_DEFAULT)
    plot_bar(df.set_index("match_id")["pass_pct"],           "Pass completion % per match (Top 20)",  "Match ID", "pedri_pass_pct_per_match.png",   top_n=TOP_N_DEFAULT)

    plot_hist(df["pressures_p90"].values,     "Pressures per 90 (hist)",     "pressures_p90",     "pedri_pressures_p90_hist.png")
    plot_hist(df["tackles_p90"].values,       "Tackles per 90 (hist)",       "tackles_p90",       "pedri_tackles_p90_hist.png")
    plot_hist(df["interceptions_p90"].values, "Interceptions per 90 (hist)", "interceptions_p90", "pedri_interceptions_p90_hist.png")

    plot_heatmap_locations(all_pedri_events, "Pedri event density (all matches)", "pedri_event_heatmap_hexbin.png")

    # --- Radars ---
    generate_pedri_radars(df)

    # --- Pass maps ---
    generate_pass_maps(all_pedri_events)

    print("✅ Done.")

if __name__ == "__main__":
    main()
