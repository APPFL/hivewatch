import argparse
import fedviz
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
            # Call this at the start of the round, after global model is sent and before client updates come in
            fedviz.round_start(round_num)


        self._last_accuracy = kwargs.get("val_accuracy", 0.0)
        self._last_loss     = kwargs.get("val_loss",     0.0)

        result = super().global_update(client_id, local_model, *args, **kwargs)

        # Log the client update after local training completes and update is received by the server

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
                        default="/home/abhijit/fedviz_clean/resources/configs/mnist/server_fedavg.yaml")
    args = parser.parse_args()

    server_agent_config = OmegaConf.load(args.config)

    #Init fedviz before anything else — this sets up the run and config in the dashboard

    fedviz.init(
        wandb_project = "my-fl-project",
        algorithm     = "FedAvg",
        config        = dict(server_agent_config.server_configs),
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
        if server_agent._current_round >= 0:
            fedviz.log_round(
                round           = server_agent._current_round,
                global_accuracy = server_agent._last_accuracy,
                global_loss     = server_agent._last_loss,
            )
        fedviz.finish()


if __name__ == "__main__":
    main()