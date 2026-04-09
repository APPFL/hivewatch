from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fedviz
import argparse
import os
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve
from fedviz.emitters import WandbEmitter, MLflowEmitter, SSEEmitter
from fedviz.geo import get_location, is_local, parse_ip
from fedviz.integrations import patch_communicator_for_geo


# ── FedViz Server Agent ───────────────────────────────────────────────────────
class FedVizServerAgent(ServerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_round = -1
        self._last_accuracy = 0.0
        self._last_loss     = 0.0
        self._peer_by_client = {}       # client_id (str) -> raw peer string
        self._client_locs    = {}       # client_id (str) -> location dict

    def global_update(self, client_id, local_model, *args, **kwargs):
        round_num = kwargs.get("round", 0)

        # ── Log round transition ──────────────────────────────────────────────
        if round_num != self._current_round:
            if self._current_round >= 0:
                fedviz.log_round(
                    round           = self._current_round,
                    global_accuracy = self._last_accuracy,
                    global_loss     = self._last_loss,
                )
            self._current_round = round_num
            fedviz.round_start(round_num)

        self._last_accuracy = kwargs.get("val_accuracy", 0.0)
        self._last_loss     = kwargs.get("val_loss",     0.0)

        # ── Resolve client IP and location ────────────────────────────────────
        raw_peer = self._peer_by_client.get(str(client_id), "")
        parsed_ip = parse_ip(raw_peer)
        geo = {}

        print(f"\n[CLIENT] client_id={client_id} | round={round_num}")
        print(f"  raw_peer  : {raw_peer!r}")
        print(f"  parsed_ip : {parsed_ip!r}")

        if parsed_ip and not is_local(parsed_ip):
            # Resolve location only once per client
            if str(client_id) not in self._client_locs:
                self._client_locs[str(client_id)] = get_location(parsed_ip)
            
            loc = self._client_locs[str(client_id)]
            if loc:
                print(f"  location  : {loc.get('city')}, {loc.get('region')}, {loc.get('country')}")
                print(f"  org       : {loc.get('org')}")
                print(f"  lat/lng   : {loc.get('lat')}, {loc.get('lng')}")
                geo = {k: v for k, v in loc.items() if k in ("lat", "lng", "city", "country")}
            else:
                print(f"  location  : Could not resolve")
        else:
            print(f"  location  : Local/non-routable, skipping geo resolution")

        # ── Run aggregation ───────────────────────────────────────────────────
        result = super().global_update(client_id, local_model, *args, **kwargs)

        # ── Log client update to fedviz ───────────────────────────────────────
        fedviz.log_client_update(
            client_id      = client_id,
            round          = round_num,
            local_accuracy = kwargs.get("val_accuracy"),
            local_loss     = kwargs.get("val_loss"),
            num_samples    = kwargs.get("num_samples", 0),
            gradient_norm  = kwargs.get("gradient_norm"),
            train_time_sec = kwargs.get("train_time_sec"),
            cpu_pct        = kwargs.get("cpu_pct"),
            ram_mb         = kwargs.get("ram_mb"),
            bytes_sent     = kwargs.get("bytes_sent", 0),
            **geo,
        )

        return result

# ── Main ──────────────────────────────────────────────────────────────────────
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/server_fedavg.yaml",
    help="Path to the configuration file.",
)
args = argparser.parse_args()

server_agent_config = OmegaConf.load(args.config)
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow_experiment   = os.environ.get("MLFLOW_EXPERIMENT", "my-fl-project-mlflow")

print(f"[fedviz] MLflow tracking URI : {mlflow_tracking_uri}")
print(f"[fedviz] MLflow experiment   : {mlflow_experiment}")

fedviz.init(
    algorithm = "FedAvg",
    config    = OmegaConf.to_container(server_agent_config.server_configs, resolve=True),
    emitters  = [
        WandbEmitter(project="my-fl-project-wandb"),
        SSEEmitter(port=7070, serve_map=True),
        MLflowEmitter(
            tracking_uri                     = mlflow_tracking_uri,
            experiment                       = mlflow_experiment,
            mlflow_system_metrics            = True,
            run_name                         = "fedavg-run-1",
            system_metrics_sampling_interval = 5,
        ),
    ],
)

server_agent = FedVizServerAgent(server_agent_config=server_agent_config)

communicator = GRPCServerCommunicator(
    server_agent,
    logger=server_agent.logger,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)
patch_communicator_for_geo(communicator, server_agent)

try:
    serve(communicator, **server_agent_config.server_configs.comm_configs.grpc_configs)
finally:
    # ── Log final round ───────────────────────────────────────────────────────
    if server_agent._current_round >= 0:
        fedviz.log_round(
            round           = server_agent._current_round,
            global_accuracy = server_agent._last_accuracy,
            global_loss     = server_agent._last_loss,
        )

    # ── Print client IP and location summary ──────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING COMPLETE — CLIENT SUMMARY")
    print("="*60)
    for client_id, raw_peer in server_agent._peer_by_client.items():
        parsed_ip = parse_ip(raw_peer)
        loc = server_agent._client_locs.get(client_id, {})
        print(f"\n  Client   : {client_id}")
        print(f"  IP       : {parsed_ip}")
        print(f"  City     : {loc.get('city', 'Unknown')}")
        print(f"  Region   : {loc.get('region', 'Unknown')}")
        print(f"  Country  : {loc.get('country', 'Unknown')}")
        print(f"  Org      : {loc.get('org', 'Unknown')}")
        print(f"  Lat/Lng  : {loc.get('lat')}, {loc.get('lng')}")
    print("="*60 + "\n")

    fedviz.finish()
