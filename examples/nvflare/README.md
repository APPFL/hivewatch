# HiveWatch Integration for NVFLARE `hello-pt`

This example shows the minimal code
changes needed to integrate `hivewatch` into NVFLARE's upstream
`examples/hello-world/hello-pt` PyTorch example.

The original FL workflow still comes from NVFLARE:

- `FedAvgRecipe` builds the job
- NVFLARE runs the server-side FedAvg workflow
- `client.py` performs local PyTorch training

The HiveWatch-specific work in this directory adds:

- server-side `hivewatch.init(...)`
- per-round `hivewatch.round_start(...)`
- per-client `hivewatch.log_client_update(...)`
- per-round aggregate `hivewatch.log_round(...)`
- run shutdown with `hivewatch.finish()`
- client geo metadata for the map UI
- emitters for W&B, MLflow, and SSE/map replay

## Files

```text
nvflare/
|-- README.md
|-- job.py            # NVFLARE recipe + HiveWatch FedAvg wrapper
|-- client.py         # NVFLARE client training script with HiveWatch metrics
|-- model.py          # PyTorch CNN
|-- requirements.txt  # extra runtime deps for this example
```

## What Changed

### `job.py`

This file adds a `HiveWatchFedAvg` class that subclasses NVFLARE's
`FedAvg` workflow and injects HiveWatch calls into the server lifecycle.

The integration points are:

- `run()`
  Initializes HiveWatch, creates emitters, records server metadata, and
  finishes the run when training ends.
- `send_model()`
  Marks the start of a round with `hivewatch.round_start(...)`.
- `_aggregate_one_result()`
  Logs client-level metrics into the shared HiveWatch run.
- `_get_aggregated_result()`
  Logs aggregated round metrics after server-side reduction.
- `enable_hivewatch(...)`
  Replaces NVFLARE's default `FedAvg` controller inside the recipe with the
  HiveWatch-aware subclass.

### `client.py`

The client remains a standard NVFLARE client script, but with a few additions:

- resolves client location with `hivewatch.geo.get_location()`
- reports local metrics in `FLModel.metrics`
- keeps `local_accuracy`, `local_loss`, `num_samples`, and `train_time_sec`
  so the server wrapper can forward them to HiveWatch
- protects CIFAR-10 download with a file lock so multiple local clients do not
  corrupt the shared dataset cache




## Install

From the repository root:

```bash
pip install -e ".[wandb,mlflow]"
pip install nvflare
pip install -r examples/nvflare/requirements.txt
```

## Runtime Configuration

This example can emit to three backends:

- `WandbEmitter`
- `MLflowEmitter`
- `SSEEmitter`

Environment variables used by `job.py`:

```bash
export WANDB_PROJECT=hivewatch-nvflare-hello-pt
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT=hivewatch-nvflare-hello-pt
export HIVEWATCH_RUN_NAME=nvflare-hello-pt
export HIVEWATCH_PORT=7070
```

Important:

- If `wandb` is enabled, you must also run `wandb login` or set `WANDB_API_KEY`.
- `MLFLOW_TRACKING_URI` must point to a running MLflow server.
- The map UI is served by `SSEEmitter` on `http://localhost:${HIVEWATCH_PORT}`.

## Run

This directory currently runs the example in NVFLARE simulation mode.

Plain NVFLARE run:

```bash
python job.py
```

HiveWatch-enabled run:

```bash
python job.py --hivewatch
```

HiveWatch plus cross-site evaluation:

```bash
python job.py --hivewatch --cross_site_eval
```

## Notes

- This example still uses CIFAR-10 for all clients, so it is an integration
  test rather than a realistic FL benchmark.
- Since this is local simulation, all clients may resolve to the same public IP
  location, so markers can overlap on the map.

