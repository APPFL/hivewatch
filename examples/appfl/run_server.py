
import fedviz
import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve
from fedviz.emitters import WandbEmitter, MLflowEmitter

class FedVizServerAgent(ServerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_round = -1
        self._last_accuracy = 0.0
        self._last_loss     = 0.0
       

    def global_update(self, client_id, local_model, *args, **kwargs):
        round_num = kwargs.get("round", 0)

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

        result = super().global_update(client_id, local_model, *args, **kwargs)

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

fedviz.init(
    algorithm = "FedAvg",
    config    = dict(server_agent_config.server_configs),
    emitters  = [
        WandbEmitter(project="my-fl-project-wandb"),
        MLflowEmitter(
            tracking_uri="http://localhost:5000",
            experiment="my-fl-project-mlflow", 
            mlflow_system_metrics=True, 
            run_name="fedavg-run-1",
            system_metrics_sampling_interval=5
        ),
    ],
)

server_agent = FedVizServerAgent(server_agent_config=server_agent_config)

communicator = GRPCServerCommunicator(
    server_agent,
    logger=server_agent.logger,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)

try:
    serve(communicator, **server_agent_config.server_configs.comm_configs.grpc_configs)
finally:
    # Log the last round before finishing
    if server_agent._current_round >= 0:
        fedviz.log_round(
            round           = server_agent._current_round,
            global_accuracy = server_agent._last_accuracy,
            global_loss     = server_agent._last_loss,
        )
    fedviz.finish()

