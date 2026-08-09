from pathlib import Path

import yaml


def test_retained_searxng_engines_are_explicitly_enabled():
    settings_path = Path(__file__).parents[1] / "searxng/config/settings.yml.template"
    settings = yaml.safe_load(settings_path.read_text())
    retained = set(settings["use_default_settings"]["engines"]["keep_only"])
    overrides = {engine["name"]: engine for engine in settings["engines"]}

    assert retained == set(overrides)
    assert all(overrides[name].get("disabled") is False for name in retained)
