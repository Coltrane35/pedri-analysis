from importlib.util import find_spec


def test_core_analysis_module_is_discoverable():
    assert find_spec("core.pedri_analysis") is not None


def test_automation_module_is_discoverable():
    assert find_spec("automation.run_all") is not None
