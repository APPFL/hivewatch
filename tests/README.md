# Tests for fedviz

Test files should take the form `*_test.py` and tests inside files should be top-level functions named `test_*()`.

Pytest discovers tests per file, so:
- run_test.py covers FedVizRun behavior
- mlflow_emitter_test.py covers MLflow emitter behavior
- wandb_emitter_test.py covers W&B emitter behavior

If you run:

- python -m pytest tests/run_test.py
it will only run that one file.

If you want all emitter-related tests, run:

- python -m pytest tests/mlflow_emitter_test.py tests/wandb_emitter_test.py

If you want everything in tests/, run:

- python -m pytest

