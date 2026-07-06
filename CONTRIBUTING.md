# Contributing

Thanks for your interest in improving this project!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

## Before opening a pull request

- Keep changes focused; one logical change per PR.
- Add or update tests for any behaviour you change — CI runs `pytest` on
  Python 3.10, 3.11, and 3.13, plus a Docker build-and-smoke-test.
- Run `pytest -q` locally and make sure it passes.
- Keep the README accurate if you change public behaviour or configuration.

## Reporting bugs

Open an issue with a minimal reproduction, the expected vs. actual behaviour,
and your environment (OS, Python version). For security issues, see
[SECURITY.md](SECURITY.md) instead of filing a public issue.
