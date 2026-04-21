from __future__ import annotations

import argparse

from .map import MapServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hivewatch")
    subparsers = parser.add_subparsers(dest="command")

    map_parser = subparsers.add_parser("map", help="Map dashboard commands")
    map_subparsers = map_parser.add_subparsers(dest="map_command")

    run_parser = map_subparsers.add_parser("run", help="Serve the map dashboard")
    run_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    run_parser.add_argument("--port", type=int, default=7070, help="Port to bind")
    run_parser.add_argument("--runs-dir", default="runs", help="Directory containing *.jsonl/*.map.json")
    run_parser.add_argument("--run-id", default=None, help="Load a specific run (by ID or filename)")
    run_parser.add_argument("--map-path", default=None, help="Optional path to fedviz_map.html")
    run_parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="How often to poll the runs directory for updates",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "map" and args.map_command == "run":
        # Disable watch mode when loading a specific run
        watch_mode = args.run_id
        
        server = MapServer(
            host=args.host,
            port=args.port,
            runs_dir=args.runs_dir,
            run_id=args.run_id,
            map_path=args.map_path,
            watch=watch_mode,
            poll_interval=args.poll_interval,
        )
        server.start()
        print(f"[hivewatch] map dashboard -> http://localhost:{args.port}")
        print(f"[hivewatch] runs directory -> {args.runs_dir}")
        if args.run_id:
            print(f"[hivewatch] loading run -> {args.run_id} (static mode)")
        else:
            print(f"[hivewatch] watching for live updates")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.stop()
        return 0

    parser.print_help()
    return 1
