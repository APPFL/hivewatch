# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code show to use NVIDIA FLARE Job Recipe to connect both Federated learning client and server algorithm
and run it under different environments
"""
import argparse
import os
import socket

from model import SimpleNetwork

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking
from nvflare.recipe.utils import add_cross_site_evaluation


class HiveWatchFedAvg(FedAvg):
    def __init__(self, *args, hivewatch_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hivewatch_config = hivewatch_config or {}
        self._hivewatch_rounds = set()
        self._hivewatch_location = {}

    def _hivewatch_round(self):
        return self.current_round + 1 if self.current_round is not None else None

    def run(self):
        import hivewatch
        from hivewatch.emitters import MLflowEmitter, SSEEmitter, WandbEmitter
        from hivewatch.geo import get_location

        self._hivewatch_location = get_location()
        hivewatch.init(
            algorithm="FedAvg / NVFlare",
            config=self.hivewatch_config,
            emitters=[
                WandbEmitter(project=os.environ.get("WANDB_PROJECT", "hivewatch-nvflare-hello-pt")),
                MLflowEmitter(
                    tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                    experiment=os.environ.get("MLFLOW_EXPERIMENT", "hivewatch-nvflare-hello-pt"),
                    run_name=os.environ.get("HIVEWATCH_RUN_NAME", "nvflare-hello-pt"),
                ),
                SSEEmitter(port=int(os.environ.get("HIVEWATCH_PORT", "7070")), serve_map=True),
            ],
        )
        hivewatch.set_server_metadata(
            host=socket.gethostname(),
            protocol="NVFlare simulation",
            **self._hivewatch_location,
        )
        try:
            super().run()
        finally:
            hivewatch.finish()

    def send_model(self, *args, **kwargs):
        import hivewatch

        round_num = self._hivewatch_round()
        if round_num is not None and round_num not in self._hivewatch_rounds:
            hivewatch.round_start(round_num)
            self._hivewatch_rounds.add(round_num)
        return super().send_model(*args, **kwargs)

    def _aggregate_one_result(self, result):
        import hivewatch

        if not result.params:
            super()._aggregate_one_result(result)
            return

        super()._aggregate_one_result(result)
        metrics = result.metrics or {}
        hivewatch.log_client_update(
            result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
            round=self._hivewatch_round(),
            local_accuracy=metrics.get("local_accuracy", metrics.get("accuracy")),
            local_loss=metrics.get("local_loss"),
            num_samples=metrics.get("num_samples"),
            train_time_sec=metrics.get("train_time_sec"),
            lat=metrics.get("lat"),
            lng=metrics.get("lng"),
            city=metrics.get("city"),
            country=metrics.get("country"),
        )

    def _get_aggregated_result(self):
        import hivewatch

        result = super()._get_aggregated_result()
        metrics = result.metrics or {}
        hivewatch.log_round(
            self._hivewatch_round(),
            global_accuracy=metrics.get("accuracy") or metrics.get("local_accuracy"),
            global_loss=metrics.get("local_loss"),
            num_selected=self._expected_count,
            num_stragglers=max(self._expected_count - self._received_count, 0),
        )
        return result


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_script", type=str, default="client.py")
    parser.add_argument("--hivewatch", action="store_true")
    parser.add_argument("--cross_site_eval", action="store_true")
    parser.add_argument(
        "--launch_external_process",
        action="store_true",
        help="Run train_script in a separate subprocess instead of in-process.",
    )
    parser.add_argument(
        "--client_memory_gc_rounds",
        type=int,
        default=0,
        help="Release model params and run GC every N rounds to keep client RSS flat. 0 = disabled.",
    )

    return parser.parse_args()


def enable_hivewatch(recipe, config):
    workflows = recipe.job._deploy_map["server"].app_config.workflows
    for workflow in workflows:
        controller = workflow.controller
        if isinstance(controller, FedAvg):
            controller_args = dict(
                num_clients=controller.num_clients,
                num_rounds=controller.num_rounds,
                start_round=controller.start_round,
                persistor_id=controller._persistor_id,
                model=controller.model,
                save_filename=controller.save_filename,
                aggregator=controller.aggregator,
                stop_cond=controller.stop_cond,
                patience=controller.patience,
                task_name=controller.task_name,
                exclude_vars=controller.exclude_vars,
                aggregation_weights=controller.aggregation_weights,
                memory_gc_rounds=controller.memory_gc_rounds,
                hivewatch_config=config,
            )
            if hasattr(controller, "enable_tensor_disk_offload"):
                controller_args["enable_tensor_disk_offload"] = controller.enable_tensor_disk_offload
            hivewatch_controller = HiveWatchFedAvg(**controller_args)
            hivewatch_controller.set_communicator(controller.communicator)
            workflow.controller = hivewatch_controller
            return
    raise RuntimeError("Could not find NVFlare FedAvg controller to wrap with HiveWatch.")


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    batch_size = args.batch_size

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=n_clients,
        num_rounds=num_rounds,
        # Model can be specified as class instance or dict config:
        model=SimpleNetwork(),
        # Alternative: model={"class_path": "model.SimpleNetwork", "args": {}},
        # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt",
        train_script=args.train_script,
        train_args=f"--batch_size {batch_size}",
        launch_external_process=args.launch_external_process,
        client_memory_gc_rounds=args.client_memory_gc_rounds,
    )
    if args.hivewatch:
        enable_hivewatch(recipe, vars(args))
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    if args.cross_site_eval:
        add_cross_site_evaluation(recipe)

    # Run FL simulation
    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    main()
