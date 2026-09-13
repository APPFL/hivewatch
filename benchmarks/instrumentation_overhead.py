#!/usr/bin/env python3
"""
Benchmark: hivewatch instrumentation overhead.

Measures per-call wall-clock latency of log_client_update() and log_round()
under two configurations:

  none  - no emitters attached (bookkeeping only: schema normalization,
          derived-metric computation, round-state tracking).
  sse   - a single SSEEmitter attached (synchronous JSONL append plus a
          full .map.json rewrite on every call). The embedded HTTP/SSE
          server is disabled (serve_map=False) so the numbers reflect file
          I/O only, not request serving.

The workload is synthetic: no model, dataset, or FL framework is involved.
Each invocation of this script is a single independent trial (its own
process, its own temp directory). Run it multiple times to capture
run-to-run variation; aggregate_overhead.py pools the resulting JSON files.

Usage:
    python instrumentation_overhead.py [--rounds 50] [--clients 20] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hivewatch
from hivewatch.emitters import SSEEmitter


def percentile(data, p):
    data = sorted(data)
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def summarize(samples_us):
    return {
        "mean":   statistics.mean(samples_us),
        "median": statistics.median(samples_us),
        "stdev":  statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
        "p95":    percentile(samples_us, 95),
        "n":      len(samples_us),
        "samples": samples_us,
    }


def run_trial(rounds, clients_per_round, emitters):
    client_update_us = []
    round_us = []

    run = hivewatch.init(algorithm="FedAvg", emitters=emitters, verbose=False)

    for r in range(rounds):
        run.round_start(r)
        for c in range(clients_per_round):
            metadata = dict(
                round=r,
                local_accuracy=0.5 + 0.001 * r,
                local_loss=1.0 - 0.001 * r,
                num_samples=1000,
                gradient_norm=0.1 + 0.01 * c,
                bytes_sent=1_000_000,
                bytes_received=900_000,
                train_time_sec=1.23,
                cpu_pct=50.0,
                ram_mb=512.0,
                lat=37.0 + c * 0.1,
                lng=-122.0 + c * 0.1,
                country="US",
            )
            t0 = time.perf_counter()
            run.log_client_update(f"client-{c}", **metadata)
            t1 = time.perf_counter()
            client_update_us.append((t1 - t0) * 1e6)

        t0 = time.perf_counter()
        run.log_round(
            r,
            global_accuracy=0.5 + 0.001 * r,
            global_loss=1.0 - 0.001 * r,
            num_selected=clients_per_round,
            num_stragglers=0,
            num_failures=0,
        )
        t1 = time.perf_counter()
        round_us.append((t1 - t0) * 1e6)

    run.finish()
    return client_update_us, round_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="hivewatch_bench_")
    try:
        results = {"rounds": args.rounds, "clients_per_round": args.clients}

        cu, r = run_trial(args.rounds, args.clients, emitters=[])
        results["none"] = {
            "log_client_update": summarize(cu),
            "log_round": summarize(r),
        }

        sse = SSEEmitter(runs_dir=os.path.join(tmp_dir, "runs"), serve_map=False)
        cu, r = run_trial(args.rounds, args.clients, emitters=[sse])
        results["sse"] = {
            "log_client_update": summarize(cu),
            "log_round": summarize(r),
        }

        out = json.dumps(results, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(out)
        print(out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
