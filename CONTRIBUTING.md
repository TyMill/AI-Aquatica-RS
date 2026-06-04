# Contributing

AI-Aquatica-RS is developed as a small, reproducible scientific Python package. Contributions should preserve deterministic behavior, explicit validation, and test coverage.

## Local workflow

```bash
python -m pip install -e .[dev]
pytest -q
python -m ai_aquatica_rs.cli validate-config --config configs/default.yaml
python -m ai_aquatica_rs.cli train-reconstruction --config configs/default.yaml
```

## Pull request checklist

- Public functions are typed and documented.
- New behavior is covered by tests.
- Errors are explicit and actionable.
- Example configuration remains executable.
- Documentation is updated when user-facing behavior changes.
