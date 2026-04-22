import os
import socket
import hivewatch
import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve
from hivewatch.emitters import WandbEmitter, MLflowEmitter, SSEEmitter
from hivewatch.geo import get_location


class HivewatchServerAgent(ServerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_round = -1
        self._last_accuracy = 0.0
        self._last_loss     = 0.0

    def global_update(self, client_id, local_model, *args, **kwargs):
        round_num = kwargs.get("round", 0)

        # Log round transition
        if round_num != self._current_round:
            if self._current_round >= 0:
                hivewatch.log_round(
                    round           = self._current_round,
                    global_accuracy = self._last_accuracy,
                    global_loss     = self._last_loss,
                )
            self._current_round = round_num
            hivewatch.round_start(round_num)

        self._last_accuracy = kwargs.get("val_accuracy", 0.0)
        self._last_loss     = kwargs.get("val_loss",     0.0)

        # Run aggregation
        result = super().global_update(client_id, local_model, *args, **kwargs)

        # Log client update to hivewatch
        hivewatch.log_client_update(
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
            lat            = kwargs.get("lat"),
            lng            = kwargs.get("lng"),
            city           = kwargs.get("city"),
            country        = kwargs.get("country"),
        )

        return result

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

print(f"[hivewatch] MLflow tracking URI : {mlflow_tracking_uri}")
print(f"[hivewatch] MLflow experiment   : {mlflow_experiment}")

hivewatch.init(
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

server_location = get_location()
server_metadata = {
    "host": socket.gethostname(),
    "protocol": "gRPC / APPFL",
    **server_location,
}
hivewatch.set_server_metadata(**server_metadata)
print(
    "[hivewatch/server] resolved server location "
    f"{server_metadata.get('city', 'Unknown')}, {server_metadata.get('country', 'Unknown')} "
    f"({server_metadata.get('lat')}, {server_metadata.get('lng')})"
)

server_agent = HivewatchServerAgent(server_agent_config=server_agent_config)

communicator = GRPCServerCommunicator(
    server_agent,
    logger=server_agent.logger,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)

try:
    serve(communicator, **server_agent_config.server_configs.comm_configs.grpc_configs)
finally:
    if server_agent._current_round >= 0:
        hivewatch.log_round(
            round           = server_agent._current_round,
            global_accuracy = server_agent._last_accuracy,
            global_loss     = server_agent._last_loss,
        )

    hivewatch.finish()
