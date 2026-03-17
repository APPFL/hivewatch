import argparse
import fedviz
from fedviz.emitters import WandbEmitter, MLflowEmitter
from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve
from omegaconf import OmegaConf


class FedVizServerAgent(ServerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_round = -1
        self._last_accuracy = 0.0
        self._last_loss     = 0.0

    def global_update(self, client_id, local_model, *args, **kwargs):
        round_num = kwargs.get("round", 0)

        # Round boundary — new round started
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

        # Call original — pass everything through untouched
        result = super().global_update(client_id, local_model, *args, **kwargs)

        # Log client update after local training completes and update is received
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="/home/abhijit/APPFL/examples/resources/configs/mnist/server_fedavg.yaml")
    args = parser.parse_args()

    server_agent_config = OmegaConf.load(args.config)

    # Init fedviz before anything else
    fedviz.init(
        algorithm = "FedAvg",
        config    = dict(server_agent_config.server_configs),
        emitters  = [
            WandbEmitter(project="my-fl-project"),
            # Add more emitters here
            MLflowEmitter(tracking_uri="http://localhost:5000"),
        ],
    )

    server_agent = FedVizServerAgent(server_agent_config=server_agent_config)

    grpc_configs = server_agent_config.server_configs.comm_configs.grpc_configs

    communicator = GRPCServerCommunicator(
        server_agent,
        logger=server_agent.logger,
        **grpc_configs,
    )

    try:
        serve(communicator, **grpc_configs)
    finally:
        # Log the last round before finishing
        if server_agent._current_round >= 0:
            fedviz.log_round(
                round           = server_agent._current_round,
                global_accuracy = server_agent._last_accuracy,
                global_loss     = server_agent._last_loss,
            )
        fedviz.finish()


if __name__ == "__main__":
    main()