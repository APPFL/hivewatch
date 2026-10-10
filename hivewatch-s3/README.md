# hivewatch-s3

`hivewatch-s3` is a clean, S3-first implementation of HiveWatch for APPFLx jobs
running inside ECS containers.

Unlike the existing local-first `fedviz` package, this implementation assumes:

- `appflx.link` launches and manages ECS tasks
- HiveWatch runs inside the training container as a library
- S3 is the canonical store for both active and completed runs
- the frontend reads map artifacts from S3 rather than from a live container port

## What it writes

For each run, the package writes local artifacts under a container-local
`runs/` directory:

- `runs/<run_id>.jsonl`
- `runs/<run_id>.map.json`

and publishes them under:

`<base_dir>/<task_id>/hivewatch/`

- `runs/<run_id>.jsonl` - full event stream for replay/debugging
- `runs/<run_id>.map.json` - map-ready payload for the frontend
- `manifest.json` - top-level run metadata and artifact pointers
- `events.jsonl` - stable alias to the latest run JSONL
- `map.json` - stable alias to the latest run map payload

## Intended integration

This package is designed for an APPFLx entry point like:

1. `appflx.link` launches an ECS task
2. the task runs the APPFL server process
3. the server process imports `hivewatch_s3`
4. client updates and round summaries are logged during training
5. artifacts are continuously uploaded to S3 during the run

## Quick example

```python
from hivewatch_s3.appflx import APPFLxTracker

tracker = APPFLxTracker.from_server_agent(
    server_agent=server_agent,
    base_dir=args.base_dir,
)

tracker.set_server_metadata(
    host="appflx-server",
    protocol="Globus Compute / APPFL",
)

while not server_agent.training_finished():
    client_id, client_model, client_metadata = (
        server_communicator.recv_result_from_one_client()
    )

    tracker.log_client_result(client_id, client_metadata)

    global_model = server_agent.global_update(
        client_id,
        client_model,
        **client_metadata,
    )

tracker.finish()
```

See `examples/appflx_entry_point_with_hivewatch.py` for a fuller integration
sketch based on the APPFLx entry point pattern.

## APPFLx runtime integration

The APPFLx entry point in the `appfl` conda environment can import this package
directly once it is installed:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate appfl
pip install --no-build-isolation -e /home/abhijit/geomap/hivewatch-s3
```

Optional runtime environment variables used by the patched APPFLx entry point:

- `HIVEWATCH_S3_BUCKET` - defaults to `appflx-bucket`
- `HIVEWATCH_ARTIFACT_DIR` - defaults to `hivewatch`
- `HIVEWATCH_UPLOAD_INTERVAL_SEC` - defaults to `1.0`
- `HIVEWATCH_SERVER_HOST` - defaults to `appflx-ecs-server`

## Real S3 smoke test

To validate the production artifact path without a full APPFL run:

```bash
cd /home/abhijit/geomap/hivewatch-s3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate appfl

PYTHONPATH=src python examples/s3_smoke_test.py \
  --bucket YOUR_BUCKET \
  --base-dir YOUR_GROUP_OR_PREFIX/experiment \
  --task-id smoke-test-task
```

This writes local cache files, a local `runs/` directory, and uploads the same
artifacts to:

- `s3://YOUR_BUCKET/YOUR_GROUP_OR_PREFIX/experiment/smoke-test-task/hivewatch/runs/run-smoketest-0001.jsonl`
- `s3://YOUR_BUCKET/YOUR_GROUP_OR_PREFIX/experiment/smoke-test-task/hivewatch/runs/run-smoketest-0001.map.json`
- `s3://YOUR_BUCKET/YOUR_GROUP_OR_PREFIX/experiment/smoke-test-task/hivewatch/manifest.json`
- `s3://YOUR_BUCKET/YOUR_GROUP_OR_PREFIX/experiment/smoke-test-task/hivewatch/events.jsonl`
- `s3://YOUR_BUCKET/YOUR_GROUP_OR_PREFIX/experiment/smoke-test-task/hivewatch/map.json`

If you want the local cache somewhere explicit, add:

```bash
--local-cache-dir /home/abhijit/geomap/hivewatch-s3/.tmp/smoke-test
```

If you want the local run files somewhere explicit, add:

```bash
--runs-dir /home/abhijit/geomap/hivewatch-s3/.tmp/smoke-test-runs
```
