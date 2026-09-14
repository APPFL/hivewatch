#!/usr/bin/env python3
"""
Pool multiple independent instrumentation_overhead.py trial results.

For each (configuration, call) pair reports:
  - pooled call-level mean/median/p95/stdev computed over every raw sample
    across all trials combined (true pooled distribution, not an average
    of per-trial summary stats)
  - run-to-run stdev: stdev of each trial's own mean (captures variation
    between independent process runs, e.g. OS scheduling/cache noise)

Usage:
    python aggregate_overhead.py trial1.json trial2.json ... trialN.json
"""
from __future__ import annotations

import json
import statistics
import sys


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


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: aggregate_overhead.py trial1.json ...", file=sys.stderr)
        sys.exit(1)

    trials = [json.load(open(p)) for p in paths]

    configs = ["none", "geo"]
    calls = ["log_client_update", "log_round"]

    total_rounds = trials[0]["rounds"] * len(trials)
    total_client_calls = trials[0]["rounds"] * trials[0]["clients_per_round"] * len(trials)
    print(f"Pooled over {len(trials)} independent trials "
          f"({trials[0]['rounds']} rounds x {trials[0]['clients_per_round']} clients each per trial)")
    print(f"total log_round samples = {total_rounds}, "
          f"total log_client_update samples = {total_client_calls}\n")

    header = f"{'call':20s} {'config':6s} {'mean(us)':>10s} {'median(us)':>11s} {'p95(us)':>10s} {'pooled_stdev':>13s} {'run-to-run_sd':>14s} {'n':>6s}"
    print(header)
    print("-" * len(header))

    for cfg in configs:
        for call in calls:
            pooled = []
            run_means = []
            for t in trials:
                s = t[cfg][call]["samples"]
                pooled.extend(s)
                run_means.append(t[cfg][call]["mean"])

            mean = statistics.mean(pooled)
            median = statistics.median(pooled)
            p95 = percentile(pooled, 95)
            pooled_stdev = statistics.stdev(pooled) if len(pooled) > 1 else 0.0
            run_to_run_stdev = statistics.stdev(run_means) if len(run_means) > 1 else 0.0

            print(f"{call:20s} {cfg:6s} {mean:10.1f} {median:11.1f} {p95:10.1f} "
                  f"{pooled_stdev:13.1f} {run_to_run_stdev:14.1f} {len(pooled):6d}")


if __name__ == "__main__":
    main()
