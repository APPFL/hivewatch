# fedviz

`fedviz` is a framework-agnostic monitoring toolkit for federated and distributed machine learning workloads. It provides a consistent interface for logging client updates, round summaries, and map-ready metadata across local experiments and larger deployments.

## Installation

`fedviz` requires Python 3.8 or later.

```bash
pip install -e .                 # core package
pip install -e ".[wandb]"        # Weights & Biases integration
pip install -e ".[mlflow]"       # MLflow integration
pip install -e ".[wandb,mlflow]" # both integrations
pip install -e ".[all]"          # all optional dependencies
```

## Quickstart

```python
import fedviz
from fedviz.emitters import WandbEmitter

fedviz.init(
    algorithm="FedAvg",
    emitters=[WandbEmitter(project="my-fl-project")],
)

for round_num in range(num_rounds):
    fedviz.round_start(round_num)

    for client_id, metadata in client_results.items():
        fedviz.log_client_update(
            client_id=client_id,
            round=round_num,
            **metadata,
        )

    fedviz.log_round(
        round=round_num,
        global_accuracy=agg_accuracy,
        global_loss=agg_loss,
    )

fedviz.finish()
```

## Emitters

`fedviz` uses a pluggable emitter model. Create one or more emitters and pass them to `fedviz.init()` to send the same run data to multiple destinations.

### Local map and deferred map metadata

```python
from fedviz.emitters import SSEEmitter

fedviz.init(
    algorithm="FedAvg",
    emitters=[SSEEmitter(port=7070, serve_map=False)],
)
```

`SSEEmitter` persists both of the following artifacts:

- `runs/<run_id>.jsonl` for the complete event history
- `runs/<run_id>.map.json` for map-ready metadata that can be loaded directly later

Serve the dashboard separately:

```bash
hivewatch map run --runs-dir runs --port 7070
```

Open one specific saved run in static mode:

```bash
hivewatch map run --runs-dir runs --run-id run-abc123
```

The bundled `examples/fedviz_map.html` viewer loads map metadata first and falls back to the JSONL-derived event history for older runs. This keeps local development and later replay workflows compatible with the same viewer.

### Package layout

The source tree groups related functionality into focused areas:

- `src/fedviz/map/` for map metadata construction and dashboard serving
- `src/fedviz/geo/` for gRPC peer parsing and geolocation helpers
- `src/fedviz/integrations/` for framework-specific integration utilities such as APPFL communicator patching

This organization keeps map and geo functionality inside the package rather than requiring example-local helper files.

### Weights & Biases

```python
from fedviz.emitters import WandbEmitter

fedviz.init(
    algorithm="FedAvg",
    emitters=[WandbEmitter(project="my-fl-project")],
)
```

### MLflow

```python
from fedviz.emitters import MLflowEmitter

# Local tracking directory (MLflow default)
fedviz.init(emitters=[MLflowEmitter(experiment="my-fl-project")])

# Remote tracking server
fedviz.init(emitters=[MLflowEmitter(
    tracking_uri="http://localhost:5000",
    experiment="my-fl-project",
)])

# MLflow system metrics
fedviz.init(emitters=[MLflowEmitter(
    experiment="my-fl-project",
    mlflow_system_metrics=True,
    system_metrics_sampling_interval=5,
)])
```

Start an MLflow server:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

To use a custom storage directory:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri ./my_custom_dir \
  --default-artifact-root ./my_custom_dir/artifacts
```

The MLflow UI is then available at `http://localhost:5000`.

### Multiple emitters

```python
from fedviz.emitters import MLflowEmitter, WandbEmitter

fedviz.init(
    algorithm="FedAvg",
    emitters=[
        WandbEmitter(project="my-fl-project"),
        MLflowEmitter(experiment="my-fl-project"),
    ],
)
```

### Custom emitters

```python
class MyEmitter:
    def on_init(self, run_id, algorithm, config): ...
    def on_round(self, summary, clients): ...
    def on_client_update(self, client): ...
    def finish(self): ...

fedviz.init(emitters=[MyEmitter()])
```

## APPFL Integration

`fedviz` integrates with APPFL by subclassing `ServerAgent` and intercepting `global_update()`. Basic metrics can be collected without modifying client code. The APPFL example uses package-provided geo utilities and integration helpers rather than a local `map_utils.py` file.

```python
from fedviz.emitters import MLflowEmitter, WandbEmitter

fedviz.init(
    algorithm="FedAvg",
    emitters=[
        WandbEmitter(project="my-fl-project"),
        MLflowEmitter(experiment="my-fl-project"),
    ],
)

class FedVizServerAgent(ServerAgent):
    def global_update(self, client_id, local_model, *args, **kwargs):
        round_num = kwargs.get("round", 0)
        result = super().global_update(client_id, local_model, *args, **kwargs)
        fedviz.log_client_update(
            client_id=client_id,
            round=round_num,
            local_accuracy=kwargs.get("val_accuracy"),
            local_loss=kwargs.get("val_loss"),
        )
        return result
```

To capture richer communication and system metrics such as bytes sent, gradient norm, CPU usage, and memory usage, include them in the metadata returned by your client trainer's `get_parameters()` implementation.

## Metadata Contract

`fedviz` defines the keys it understands, but it preserves unknown keys so applications can attach additional metadata without losing information.

| Field | Type | Description |
|---|---|---|
| `client_id` | str | Client identifier |
| `round` | int | Current global round |
| `local_accuracy` | float | Accuracy after local training |
| `local_loss` | float | Loss after local training |
| `num_samples` | int | Local dataset size |
| `gradient_norm` | float | L2 norm of local gradients |
| `bytes_sent` | int | Bytes uploaded to the server |
| `train_time_sec` | float | Local training wall-clock time |
| `cpu_pct` | float | CPU utilization percentage |
| `ram_mb` | float | Memory usage in MB |
| `gpu_util_pct` | float | GPU utilization percentage |
| `lat` / `lng` / `country` | float/str | Client location metadata for map visualization |
| `base_round` | int | For asynchronous FL, staleness is `round - base_round` |

## Logged Metrics

### Weights & Biases

| Metric | Description |
|---|---|
| `round/accuracy`, `round/loss` | Global model performance per round |
| `round/participation_rate` | Completed clients divided by selected clients |
| `round/num_stragglers` | Number of stragglers |
| `round/duration_sec` | Wall-clock time per round |
| `comm/total_bytes_mb` | Total upload and download volume |
| `comm/bytes_per_client_mb` | Per-client communication cost |
| `agg/gradient_divergence` | Standard deviation of per-client gradient norms |
| `agg/aggregation_time_sec` | Server-side aggregation time |
| `client/<id>/accuracy` | Per-client accuracy |
| `client/<id>/gradient_norm` | Per-client gradient norm |
| `client/<id>/staleness` | Rounds behind the current global model in async FL |
| `client/<id>/bytes_sent_mb` | Per-client upload size |
| `client/<id>/train_time_sec` | Per-client training time |
| `sys/<id>/cpu_pct` | Per-client CPU utilization |
| `sys/<id>/ram_mb` | Per-client RAM usage |
| `event/client_dropout` | Dropout counter |
| `event/comm_failure` | Communication failure counter |

All metrics use `round` as the x-axis through `wandb.define_metric()`.

### MLflow

MLflow records the same metrics. Per-client metrics use dot notation such as `client.<id>.accuracy` instead of slash notation because of MLflow metric naming conventions. Hyperparameters are logged once as MLflow parameters, and model checkpoints are stored as versioned MLflow artifacts.

## Architecture

```text
FL Clients
  └── return metadata dict
        │  (gRPC / HTTP / sockets / others; fedviz does not depend on the transport layer)
        ▼
FL Server
  └── receives metadata and calls fedviz:
        fedviz.round_start(round)
        fedviz.log_client_update(client_id, round, **metadata)
        fedviz.log_round(round, global_accuracy, global_loss)
        │
        ▼
fedviz
  ├── WandbEmitter  →  wandb.ai dashboard
  └── MLflowEmitter →  MLflow UI (localhost:5000)
```

`fedviz` does not depend on a specific transport layer or FL framework. Applications bridge their training framework to `fedviz` in the same way they would bridge it to another experiment tracking backend.

For map visualization, the storage contract includes a standalone metadata artifact in addition to the raw event log. This supports:

- local CLI runs that immediately launch or serve a map
- local or remote services that persist metadata for later display
- future deployments that store metadata in object storage and load it in a separate web tier

## Project Structure

```text
src/fedviz/
  __init__.py          public API (init, round_start, log_client_update, log_round, finish)
  __main__.py          module entrypoint
  _state.py            global singleton state
  cli.py               hivewatch CLI
  geo.py               geolocation utilities
  map_metadata.py      map metadata assembly and replay helpers
  map_server.py        local dashboard and map HTTP server
  run.py               FedVizRun class and init()
  schema.py            metadata contract (ClientUpdate, RoundSummary)
  emitters/
    __init__.py        emitter exports
    sse_emitter.py     local SSE and map metadata emitter
    wandb_emitter.py   Weights & Biases integration
    mlflow_emitter.py  MLflow integration
examples/
  appfl/               APPFL example with server and client scripts
  flwr-demo/           Flower integration example
  fedviz_map.html      bundled local map viewer
```
