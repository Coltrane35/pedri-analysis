from pathlib import Path

def test_expected_directories_exist():
    for d in ["core", "automation", "utils", ".github/workflows"]:
        assert Path(d).exists(), f"{d} should exist"
