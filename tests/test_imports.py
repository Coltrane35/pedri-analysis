import importlib.util as ilu

MODULES = [
    "core.pedri_analysis",
    "core.pedri_analysis_extended",
    "core.find_pedri_matches_in_events",
    "core.pedri_inspect_lineups",
    "core.pedri_profile",
    "automation.run_all",
    "utils.data_loader",
]

def test_core_modules_discoverable():
    for mod in MODULES:
        assert ilu.find_spec(mod) is not None, f"Cannot find spec for {mod}"
