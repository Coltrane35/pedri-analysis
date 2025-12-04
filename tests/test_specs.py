import importlib.util as ilu

MODULES = [
    "core.pedri_analysis",
    "core.pedri_analysis_extended",
    "core.pedri_inspect_lineups",
    "core.find_pedri_matches_in_events",
    "automation.run_all",
]

def test_modules_are_discoverable():
    for mod in MODULES:
        assert ilu.find_spec(mod) is not None, f"Cannot find spec for {mod}"
