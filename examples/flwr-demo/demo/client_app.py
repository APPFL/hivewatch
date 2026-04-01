"""demo: A Flower / PYTORCH app."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import io
import pickle

from demo.task import (
    Net,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        import io
        import numpy as np

        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        loss, accuracy = test(self.net, self.valloader, self.device)

        # Measure bytes sent — serialize weights to measure payload size
        weights = get_weights(self.net)
        buf = io.BytesIO()
        pickle.dump(weights, buf)
        bytes_sent = buf.tell()

        return weights, len(self.trainloader.dataset), {
            "train_loss":     train_loss,
            "local_loss":     loss,
            "local_accuracy": accuracy,
            "bytes_sent":     bytes_sent,
        }

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
