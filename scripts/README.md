# Operational scripts

These scripts provide bounded integration checks and are exposed through
`just` recipes. They do not implement the evaluation harness.

- `lfm_smoke.py`: verifies the OpenAI-compatible planning and cited-digest
  contract for a configured preprocessing endpoint.
