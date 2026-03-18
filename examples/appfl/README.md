# `APPFL` + `fedviz` Example

This example runs a 2-client federated learning job using [APPFL](https://github.com/APPFL/APPFL) with FedAvg over gRPC, while logging training metrics to W&B and MLflow via **fedviz**.

## What the example does

- **Server** (`run_server.py`): Subclasses APPFL's `ServerAgent`, intercepts each client update, and forwards metrics (accuracy, loss, gradient norm, resource usage, bytes sent) to fedviz. At the end of every round it logs aggregated round-level metrics. Uses `FedAvgAggregator` with 2 clients and runs for 10 global epochs.
- **Clients** (`run_client.py`): Standard APPFL clients. Each loads a partitioned (non-IID) split of MNIST, trains a small CNN locally with Adam, and pushes updates to the server over gRPC.

## Prerequisites

Make sure you have `appfl` and `fedviz` installed, along with the required dependencies for this example (e.g. `wandb`, `mlflow`)

## Running the example

The server must be started before the clients. Open **three terminals**, all from the `examples/appfl/` directory.

**Terminal 1 — server**
```bash
python run_server.py
# Optional: use a custom server config
python run_server.py --config ./resources/configs/server_fedavg.yaml
```

**Terminal 2 — client 1**
```bash
python run_client.py --config ./resources/configs/client_1.yaml
```

**Terminal 3 — client 2**
```bash
python run_client.py --config ./resources/configs/client_2.yaml
```

Training finishes automatically after 10 global rounds. Results are written to `./output/`.

## Monitoring backends

The server initializes two emitters in `run_server.py`:

```python
fedviz.init(
    algorithm = "FedAvg",
    config    = dict(server_agent_config.server_configs),
    emitters  = [
        WandbEmitter(project="my-fl-project-wandb"),
        MLflowEmitter(experiment="my-fl-project-mlflow"),
    ],
)
```

- **W&B**: Metrics appear in the `my-fl-project-wandb` project. Requires `WANDB_API_KEY` to be set, or run `wandb login` first.
- **MLflow**: Runs are recorded in the `my-fl-project-mlflow` experiment. The tracking URI defaults to `./mlruns`; override with `MLFLOW_TRACKING_URI`.

To use only one backend, remove the unwanted emitter from the list.
