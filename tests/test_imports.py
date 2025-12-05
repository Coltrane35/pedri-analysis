<<<<<<< HEAD
﻿import importlib.util as ilu

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
=======
﻿import importlib
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
>>>>>>> d8071b3 (test(ci): add minimal smoke tests and enable pytest in CI)
