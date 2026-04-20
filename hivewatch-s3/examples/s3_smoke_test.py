#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hivewatch_s3 import S3ArtifactEmitter, init


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real S3 smoke test for hivewatch-s3.",
    )
    parser.add_argument("--bucket", required=True, help="Target S3 bucket")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Artifact prefix root, for example group-id/experiment",
    )
    parser.add_argument(
        "--task-id",
        default="smoke-test-task",
        help="Task id to use in the S3 prefix and local cache directory",
    )
    parser.add_argument(
        "--artifact-dirname",
        default="hivewatch",
        help="Artifact folder name under <base-dir>/<task-id>/",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for the boto3 S3 client",
    )
    parser.add_argument(
        "--local-cache-dir",
        default=None,
        help="Optional local artifact cache directory",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Optional local runs directory. Defaults to <local-cache-dir>/runs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    emitter = S3ArtifactEmitter(
        base_dir=args.base_dir,
        bucket_name=args.bucket,
        task_id=args.task_id,
        artifact_dirname=args.artifact_dirname,
        aws_region=args.region,
        local_cache_dir=args.local_cache_dir,
        runs_dir=args.runs_dir,
        upload_interval_sec=0.0,
        upload_events=True,
    )

    run = init(
        run_id="run-smoketest-0001",
        algorithm="FedAvg",
        config={"mode": "s3-smoke-test"},
        emitters=[emitter],
        verbose=True,
    )

    run.set_server_metadata(
        host="smoke-test-host",
        protocol="APPFLx / HiveWatch",
    )
    run.round_start(1)
    run.log_client_update(
        "client-a",
        round=1,
        local_accuracy=0.81,
        local_loss=0.42,
        num_samples=100,
        bytes_sent=12345,
        train_time_sec=4.8,
        lat=41.8781,
        lng=-87.6298,
        city="Chicago",
        country="US",
    )
    run.log_client_update(
        "client-b",
        round=1,
        local_accuracy=0.78,
        local_loss=0.47,
        num_samples=120,
        bytes_sent=15678,
        train_time_sec=5.1,
        lat=40.7128,
        lng=-74.0060,
        city="New York",
        country="US",
    )
    run.log_round(
        1,
        global_accuracy=0.83,
        global_loss=0.39,
    )
    run.finish()

    print("\nLocal artifacts:")
    print(f" - {emitter.events_path}")
    print(f" - {emitter.map_path}")
    print(f" - {emitter.manifest_path}")

    print("\nS3 artifacts:")
    print(f" - s3://{args.bucket}/{emitter.runs_prefix}/{emitter.events_path.name}")
    print(f" - s3://{args.bucket}/{emitter.runs_prefix}/{emitter.map_path.name}")
    print(f" - s3://{args.bucket}/{emitter.artifact_prefix}/manifest.json")
    print(f" - s3://{args.bucket}/{emitter.artifact_prefix}/events.jsonl")
    print(f" - s3://{args.bucket}/{emitter.artifact_prefix}/map.json")

    manifest = json.loads(Path(emitter.manifest_path).read_text(encoding="utf-8"))
    print("\nManifest preview:")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
