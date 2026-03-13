"""
Run:
    pip install fedviz wandb && wandb login
    python examples/vanilla.py
"""
import random
import statistics
import fedviz

# ── 1. Init ───────────────────────────────────────
fedviz.init(
    wandb_project = "fedviz-demo",
    algorithm     = "FedAvg",
    config        = {"num_clients": 10, "rounds": 20, "lr": 0.01},
)

NUM_ROUNDS  = 20
NUM_CLIENTS = 10

# Simulated client registry — in real FL, clients are remote
CLIENTS = {f"client-{i:02d}": {"data_size": random.randint(300, 800)} for i in range(NUM_CLIENTS)}

global_acc  = 0.50
global_loss = 1.20

for rnd in range(NUM_ROUNDS):

    # ── 2. Round start — enables wall-time tracking ───────────────────────────
    fedviz.round_start(rnd)

    selected = random.sample(list(CLIENTS.keys()), 6)

    # ── Your aggregation logic here ─────────────────
    local_losses = []
    for client_id in selected:

        # Simulate what a client returns (metadata dict).
        # In real FL this comes over gRPC, HTTP, sockets — doesn't matter.
        # The user decides how to transport it; fedviz just reads the dict.
        metadata = {
            "round":          rnd,
            "local_accuracy": global_acc  + random.uniform(-0.05, 0.10),
            "local_loss":     global_loss * random.uniform(0.85, 1.05),
            "num_samples":    CLIENTS[client_id]["data_size"],
            "gradient_norm":  random.uniform(0.8, 2.5),
            "sparsity":       random.uniform(0.05, 0.30),
            "bytes_sent":     random.randint(2_000_000, 8_000_000),
            "train_time_sec": random.uniform(5, 30),
            "cpu_pct":        random.uniform(40, 95),
            "ram_mb":         random.uniform(512, 4096),
            "gpu_util_pct":   random.uniform(50, 99),
            "lat":     random.choice([37.7, 51.5, 35.6, 48.8, -33.8]),
            "lng":     random.choice([-122.4, -0.12, 139.6, 2.35, 151.2]),
            "country": random.choice(["US", "UK", "JP", "FR", "AU"]),
        }
        local_losses.append(metadata["local_loss"])

        # ── 3. Log each client update — pass the metadata dict straight in ────
        fedviz.log_client_update(client_id=client_id, **metadata)

    # Your aggregation 
    global_acc  = min(0.99, global_acc  + random.uniform(0.01, 0.04))
    global_loss = max(0.01, global_loss * random.uniform(0.88, 0.97))

    # ── 4. Log round summary ──────────────────────────────────────────────────
    # fedviz auto-computes: gradient_divergence, total_bytes, round_duration
    fedviz.log_round(
        round           = rnd,
        global_accuracy = global_acc,
        global_loss     = global_loss,
        num_stragglers  = random.randint(0, 1),
        algorithm_metadata = {"lr": 0.01},
    )

# ── 5. Finish ─────────────────────────────────────────────────────────────────
fedviz.finish()
