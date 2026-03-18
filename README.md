# fedviz

Framework-agnostic monitoring toolkit for federated and distributed ML training. 

## Installation

> 💡  `fedviz` requires `Python >= 3.8`

```bash
pip install -e .                     # core only, zero dependencies
pip install -e ".[wandb]"            # + W&B integration
pip install -e ".[mlflow]"           # + MLflow integration
pip install -e ".[wandb,mlflow]"     # both
pip install -e ".[all]"              # everything
```

## Quickstart

```python
import fedviz
from fedviz.emitters import WandbEmitter

fedviz.init(
    algorithm = "FedAvg",
    emitters  = [WandbEmitter(project="my-fl-project")],
)

for round_num in range(num_rounds):
    fedviz.round_start(round_num)

    # your training loop — any framework
    for client_id, metadata in client_results.items():
        fedviz.log_client_update(
            client_id=client_id, 
            round=round_num, 
            **metadata
        )

    fedviz.log_round(
        round=round_num,
        global_accuracy=agg_accuracy,
        global_loss=agg_loss,
    )

fedviz.finish()
```

## Emitters

`fedviz` uses a pluggable emitter system. Construct emitter instances and pass them to `init()`. You can use multiple emitters simultaneously.

**Weights & Biases:**

```python
from fedviz.emitters import WandbEmitter

fedviz.init(
    algorithm = "FedAvg",
    emitters  = [WandbEmitter(project="my-fl-project")],
)
```

**MLflow:**

```python
from fedviz.emitters import MLflowEmitter

# Local — logs to ./mlruns, view with command: mlflow ui
fedviz.init(emitters=[MLflowEmitter(experiment="my-fl-project")])

# Remote server — start with command:
# mlflow server --host 0.0.0.0 --port 5000
fedviz.init(emitters=[MLflowEmitter(
    tracking_uri = "http://localhost:5000",
    experiment   = "my-fl-project",
)])

# Enable MLflow's built-in server-side system metrics (CPU, RAM, GPU, disk, network)
# sampled in the background and visible in the MLflow UI "System Metrics" tab
fedviz.init(emitters=[MLflowEmitter(
    experiment             = "my-fl-project",
    mlflow_system_metrics  = True,
    system_metrics_sampling_interval = 5,  # seconds; default is 10
)])
```

**Both simultaneously:**

```python
from fedviz.emitters import WandbEmitter, MLflowEmitter

fedviz.init(
    algorithm = "FedAvg",
    emitters  = [
        WandbEmitter(project="my-fl-project"),
        MLflowEmitter(experiment="my-fl-project"),
    ],
)
```

**Custom emitter — implement these hooks:**

```python
class MyEmitter:
    def on_init(self, run_id, algorithm, config): ...
    def on_round(self, summary, clients):         ...
    def on_client_update(self, client):           ...
    def finish(self):                             ...

fedviz.init(emitters=[MyEmitter()])
```

## APPFL Integration

fedviz works with APPFL by subclassing `ServerAgent` to intercept `global_update()`. No changes to client code are required for basic metrics. See `examples/appfl_server.py` for the full server script.

```python
from fedviz.emitters import WandbEmitter, MLflowEmitter

fedviz.init(
    algorithm = "FedAvg",
    emitters  = [
        WandbEmitter(project="my-fl-project"),
        MLflowEmitter(experiment="my-fl-project"),
    ],
)

class FedVizServerAgent(ServerAgent):
    def global_update(self, client_id, local_model, *args, **kwargs):
        round_num = kwargs.get("round", 0)
        result = super().global_update(client_id, local_model, *args, **kwargs)
        fedviz.log_client_update(
            client_id      = client_id,
            round          = round_num,
            local_accuracy = kwargs.get("val_accuracy"),
            local_loss     = kwargs.get("val_loss"),
        )
        return result
```

For richer communication metrics (bytes sent, gradient norm, CPU/RAM), add them to your client trainer's `get_parameters()` return metadata. See the metadata contract below.

## Metadata contract

`fedviz` defines what keys it understands. Clients fill what they have. Unknown keys are preserved and never dropped.

| Field | Type | Description |
|---|---|---|
| `client_id` | str | required |
| `round` | int | current global round |
| `local_accuracy` | float | accuracy after local training |
| `local_loss` | float | loss after local training |
| `num_samples` | int | local dataset size |
| `gradient_norm` | float | L2 norm of local gradients |
| `bytes_sent` | int | bytes uploaded to server |
| `train_time_sec` | float | local training wall time |
| `cpu_pct` | float | CPU utilisation % |
| `ram_mb` | float | RAM usage in MB |
| `gpu_util_pct` | float | GPU utilisation % |
| `lat` / `lng` / `country` | float/str | client geo for map viz |
| `base_round` | int | async FL — staleness = round - base_round |

## What gets logged

### Weights & Biases

| Metric | Description |
|---|---|
| `round/accuracy`, `round/loss` | Global model performance per round |
| `round/participation_rate` | Completed / selected clients |
| `round/num_stragglers` | Straggler count |
| `round/duration_sec` | Wall time per round |
| `comm/total_bytes_mb` | Total upload + download volume |
| `comm/bytes_per_client_mb` | Per-client communication cost |
| `agg/gradient_divergence` | Std dev of per-client gradient norms — non-IID signal |
| `agg/aggregation_time_sec` | Server aggregation compute time |
| `client/<id>/accuracy` | Per-client accuracy |
| `client/<id>/gradient_norm` | Per-client gradient norm |
| `client/<id>/staleness` | Async FL — rounds behind global model |
| `client/<id>/bytes_sent_mb` | Per-client upload size |
| `client/<id>/train_time_sec` | Per-client training time |
| `sys/<id>/cpu_pct` | Per-client CPU utilisation |
| `sys/<id>/ram_mb` | Per-client RAM usage |
| `event/client_dropout` | Dropout counter |
| `event/comm_failure` | Communication failure counter |

All metrics use `round` as the x-axis via `wandb.define_metric()`.

### MLflow

Same metrics as above. Per-client metrics use dot notation (`client.<id>.accuracy`) instead of slash due to MLflow's key format. Hyperparameters are logged once as MLflow params. Model checkpoints are logged as versioned MLflow artifacts.

## Architecture

```
FL Clients
  └── return metadata dict
        │  (gRPC / HTTP / sockets / others — fedviz never touches the data transport)
        ▼
FL Server
  └── receives metadata, calls fedviz:
        fedviz.round_start(round)
        fedviz.log_client_update(client_id, round, **metadata)
        fedviz.log_round(round, global_accuracy, global_loss)
        │
        ▼
fedviz
  ├── WandbEmitter  →  wandb.ai dashboard
  └── MLflowEmitter →  MLflow UI (localhost:5000)
```

fedviz never touches the transport layer or the framework. The user bridges their framework to fedviz the same way they would bridge it to W&B.

## Project structure

```
src/fedviz/
  __init__.py          — public API (init, round_start, log_client_update, log_round, finish)
  run.py               — FedVizRun class and init()
  schema.py            — metadata contract (ClientUpdate, RoundSummary)
  _state.py            — global singleton state
  emitters/
    __init__.py        — exports WandbEmitter, MLflowEmitter
    wandb_emitter.py   — W&B integration
    mlflow_emitter.py  — MLflow integration
examples/
  appfl_server.py      — APPFL + fedviz server
  vanilla.py           — pure Python example
```
