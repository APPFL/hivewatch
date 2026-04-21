import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator
from fedviz.geo import get_location

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/client_2.yaml",
    help="Path to the configuration file.",
)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(args.config)

client_agent = ClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id=client_agent.get_id(),
    logger=client_agent.logger,
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

init_global_model = client_communicator.get_global_model(init_model=True)
client_agent.load_parameters(init_global_model)

# Send the number of local data to the server
sample_size = client_agent.get_sample_size()
client_communicator.invoke_custom_action(
    action="set_sample_size", sample_size=sample_size
)

location = get_location()
client_geo = {
    key: location.get(key)
    for key in ("lat", "lng", "city", "country")
    if location.get(key) is not None
}

if client_geo:
    print(
        "[fedviz/client] resolved location "
        f"{client_geo.get('city', 'Unknown')}, {client_geo.get('country', 'Unknown')} "
        f"({client_geo.get('lat')}, {client_geo.get('lng')})"
    )
else:
    print("[fedviz/client] location resolution failed; sending training metrics without geo")

while True:
    client_agent.train()
    local_model = client_agent.get_parameters()
    if isinstance(local_model, tuple):
        local_model, metadata = local_model[0], local_model[1]
    else:
        metadata = {}
    metadata.update(client_geo)
    new_global_model, metadata = client_communicator.update_global_model(
        local_model, **metadata
    )
    if metadata["status"] == "DONE":
        break
    if "local_steps" in metadata:
        client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
    client_agent.load_parameters(new_global_model)
client_communicator.invoke_custom_action(action="close_connection")
