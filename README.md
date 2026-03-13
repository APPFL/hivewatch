# fedviz

Framework-agnostic monitoring toolkit for federated and distributed ML training. Designed with the same philosophy as Weights & Biases — you call it, it logs. fedviz never touches your framework.

## Why fedviz

Vanilla W&B works great for centralized training but misses the metrics that matter in federated learning — per-client gradient divergence, communication costs, participation rates, staleness in async FL, and non-IID signals. fedviz adds all of these on top of W&B without requiring any changes to how you run your framework.

## Install

```bash
pip install -e .              # core only, zero dependencies
pip install -e ".[wandb]"     # + W&B integration
```

Requires Python >= 3.8.

## Quickstart

```python
import fedviz

fedviz.init(wandb_project="my-fl-project", algorithm="FedAvg")

for round_num in range(num_rounds):
    fedviz.round_start(round_num)

    # your training loop — any framework
    for client_id, metadata in client_results.items():
        fedviz.log_client_update(client_id=client_id, round=round_num, **metadata)

    fedviz.log_round(
        round           = round_num,
        global_accuracy = agg_accuracy,
        global_loss     = agg_loss,
    )

fedviz.finish()
```

## APPFL Integration

fedviz works with APPFL by subclassing `ServerAgent` to intercept `global_update()`. No changes to your client code are required for basic metrics. See `examples/appfl_server.py` for the full server script.

```python
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

fedviz defines what keys it understands. Clients fill what they have. Unknown keys are preserved and never dropped.

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
| `base_round` | int | async FL — which round this update is based on |

## What gets logged to W&B

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

All metrics use `round` as the x-axis.

## Architecture

```
FL Clients
  └── return metadata dict 
        │  (transported via gRPC / HTTP / sockets — fedviz doesn't touch this)
        ▼
FL Server
  └── receives metadata, calls fedviz:
        fedviz.round_start(round)
        fedviz.log_client_update(client_id, round, **metadata)
        fedviz.log_round(round, global_accuracy, global_loss)
        │
        ▼
fedviz
  └── WandbEmitter
        └── wandb.log() → wandb.ai dashboard
```

fedviz never touches the transport layer or the framework. The user bridges their framework to fedviz the same way they would bridge it to W&B.

## Custom emitters

Implement `.emit()` to send metrics anywhere:

```python
class MyEmitter:
    def on_round(self, summary, clients):
        # send to your own backend
        pass
    def on_client_update(self, client):
        pass
    def finish(self):
        pass

fedviz.init(emitters=[MyEmitter()])
```

## Project structure

```
fedviz/src/fedviz
  schema.py              — metadata contract (ClientUpdate, RoundSummary)
  __init__.py            — public API (init, round_start, log_client_update, log_round, finish)
  emitters/
    wandb_emitter.py     — W&B integration (only file that imports wandb)
examples/
  appfl_server.py        — APPFL + fedviz server
  vanilla.py             — pure Python example
```