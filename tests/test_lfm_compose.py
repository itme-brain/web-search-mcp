from pathlib import Path


def test_lfm_sidecar_is_pinned_internal_and_cpu_only():
    compose = Path("docker-compose.yml").read_text()
    service = compose.split("\n  lfm:\n", 1)[1].split("\n  searxng:\n", 1)[0]

    assert 'profiles: ["lfm"]' in service
    assert "server-b10326@sha256:" in service
    assert "LiquidAI/LFM2.5-2.6B-GGUF" in service
    assert "LFM2.5-2.6B-Q8_0.gguf" in service
    assert "- --device\n      - none" in service
    assert '- --n-gpu-layers\n      - "0"' in service
    assert "ports:" not in service
    assert "lfm-models:/models" in service
