from pathlib import Path

def test_env_template_present():
    assert Path(".env.example").exists()
