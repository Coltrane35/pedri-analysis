from pathlib import Path

def test_flake8_config_present_and_valid():
    p = Path(".flake8")
    assert p.exists(), ".flake8 missing"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    assert "[flake8]" in txt

def test_black_config_present():
    p = Path("pyproject.toml")
    assert p.exists(), "pyproject.toml missing"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    assert "[tool.black]" in txt
    assert "line-length" in txt
