from pathlib import Path

import yaml


def test_retained_searxng_engines_have_explicit_activation_policy():
    settings_path = Path(__file__).parents[1] / "searxng/config/settings.yml.template"
    settings = yaml.safe_load(settings_path.read_text())
    retained = set(settings["use_default_settings"]["engines"]["keep_only"])
    overrides = {engine["name"]: engine for engine in settings["engines"]}

    assert retained == set(overrides)
    assert all(isinstance(overrides[name].get("disabled"), bool) for name in retained)
    assert overrides["bing"]["disabled"] is False
    assert overrides["wikipedia"]["disabled"] is False
    assert overrides["github"]["disabled"] is False
    assert overrides["arxiv"]["disabled"] is False
    assert overrides["duckduckgo"]["disabled"] is True
    assert overrides["startpage"]["disabled"] is True
    assert overrides["mojeek"]["disabled"] is True
