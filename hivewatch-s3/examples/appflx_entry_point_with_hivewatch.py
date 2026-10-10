"""
Example APPFLx integration sketch for hivewatch-s3.

This is intentionally kept as a reference example rather than a drop-in
replacement for the APPFL source file.
"""

from hivewatch_s3.appflx import APPFLxTracker


def install_hivewatch(server_agent, args):
    tracker = APPFLxTracker.from_server_agent(
        server_agent=server_agent,
        base_dir=args.base_dir,
        bucket_name="appflx-bucket",
        artifact_dirname="hivewatch",
        upload_interval_sec=1.0,
    )
    tracker.set_server_metadata(
        host="appflx-ecs-server",
        protocol="Globus Compute / APPFL",
    )
    return tracker


def training_loop_example(server_agent, server_communicator, tracker):
    server_communicator.send_task_to_all_clients(task_name="train")
    round_counters = {}

    while not server_agent.training_finished():
        client_id, client_model, client_metadata = (
            server_communicator.recv_result_from_one_client()
        )

        round_num = int(client_metadata.get("round", round_counters.get(client_id, 0)))
        tracker.round_start(round_num)
        tracker.log_client_result(client_id, client_metadata)

        global_model = server_agent.global_update(
            client_id,
            client_model,
            **client_metadata,
        )

        round_counters[client_id] = round_num + 1

        if not server_agent.training_finished():
            tracker.log_round(
                round_num,
                global_accuracy=client_metadata.get("global_accuracy"),
                global_loss=client_metadata.get("global_loss"),
            )

    tracker.finish()
