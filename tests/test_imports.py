import importlib
import pathlib

def test_core_modules_importable():
    modules = [
        "core.pedri_analysis",
        "core.pedri_analysis_extended",
        "core.find_pedri_matches_in_events",
        "core.pedri_inspect_lineups",
        "core.pedri_profile",
        "automation.run_all",
        "utils.data_loader",
    ]
    for mod in modules:
        importlib.import_module(mod)
