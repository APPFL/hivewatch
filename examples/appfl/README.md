# `APPFL` + `fedviz` Example

This example runs a 2-client federated learning job using [APPFL](https://github.com/APPFL/APPFL) with FedAvg over gRPC, while logging training metrics and map metadata via **fedviz**.

## What the example does

- **Server** (`run_server.py`): Subclasses APPFL's `ServerAgent`, intercepts each client update, and forwards metrics (accuracy, loss, gradient norm, resource usage, bytes sent) to fedviz. At the end of every round it logs aggregated round-level metrics. It now imports geo helpers from `fedviz.geo` and communicator patching helpers from `fedviz.integrations` instead of relying on a local `map_utils.py`. Uses `FedAvgAggregator` with 2 clients and runs for 10 global epochs.
- **Clients** (`run_client.py`): Standard APPFL clients. Each loads a partitioned (non-IID) split of MNIST, trains a small CNN locally with Adam, and pushes updates to the server over gRPC.

## Prerequisites

Make sure you have `appfl` and `fedviz` installed, along with the required dependencies for this example (e.g. `wandb`, `mlflow`)

## Running the example

The server must be started before the clients. Open **four terminals**, all from the `examples/appfl/` directory.

**Terminal 1 — server**
```bash
python run_server.py
# Optional: use a custom server config
python run_server.py --config ./resources/configs/server_fedavg.yaml
```

**Terminal 2 — map**
```bash
hivewatch map run --runs-dir runs --port 7070
```

**Terminal 3 — client 1**
```bash
python run_client.py --config ./resources/configs/client_1.yaml
```

**Terminal 4 — client 2**
```bash
python run_client.py --config ./resources/configs/client_2.yaml
```

Training finishes automatically after 10 global rounds. Results are written to `./output/`.

## Monitoring backends

The server initializes three emitters in `run_server.py`:

```python
fedviz.init(
    algorithm = "FedAvg",
    config    = dict(server_agent_config.server_configs),
    emitters  = [
        WandbEmitter(project="my-fl-project-wandb"),
        SSEEmitter(port=7070, serve_map=False),
        MLflowEmitter(experiment="my-fl-project-mlflow"),
    ],
)
```

- **W&B**: Metrics appear in the `my-fl-project-wandb` project. Requires `WANDB_API_KEY` to be set, or run `wandb login` first.
- **MLflow**: Runs are recorded in the `my-fl-project-mlflow` experiment. The tracking URI defaults to `./mlruns`; override with `MLFLOW_TRACKING_URI`.
- **Map metadata + local viewer**: `SSEEmitter` writes raw events to `runs/<run_id>.jsonl` and map-ready metadata to `runs/<run_id>.map.json`. Serve the dashboard separately with `hivewatch map run --runs-dir runs --port 7070`.

To use only one backend, remove the unwanted emitter from the list.

For an Appflx-style deployment, the map metadata file is the preferred interchange format: save it locally during training, move it to shared storage such as S3, and have a separate web tier reload it later with the same viewer.

## Source layout

The APPFL example now uses the package layout below:

- `src/fedviz/map/` for map metadata and map server code
- `src/fedviz/geo/` for IP parsing and geolocation helpers
- `src/fedviz/integrations/` for APPFL communicator patch helpers
