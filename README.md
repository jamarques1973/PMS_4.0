# PMS Orchestrator (Python)

CLI orchestration for data loading, model training, hyperparameter optimization, evaluation, and persistence. Multi-engine HPO support (none, random, grid, Optuna, Keras Tuner).

## Quickstart

1) Create a CSV `sample.csv` with columns including target `y`.
2) Edit `configs/example_svr.yaml` to point to your data.
3) Run without installing the package using the module path:
```bash
python -m pms.cli --help
python -m pms.cli show-config --config configs/example_svr.yaml
python -m pms.cli run --config configs/example_svr.yaml
```

Artifacts and metadata will be stored under `train.output_dir`.

## Design

- Python-only orchestrator (`click` CLI)
- Modular trainers (`pms/training`)
- Pluggable HPO engines (`pms/hpo`)
- Config-driven runs (YAML)