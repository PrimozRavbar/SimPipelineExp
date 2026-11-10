"""python -m console [--port 8000] [--seed 42] [--users-per-tick 10] [--world static|dynamic]"""

import argparse

from console.server import serve


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m console", description="SimPipeline live console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users-per-tick", type=int, default=10)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--world", choices=["static", "dynamic"], default="static",
                        help="dynamic: taste drift, users joining, fatigue, no rewatching")
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, seed=args.seed,
          users_per_tick=args.users_per_tick, runs_dir=args.runs_dir, world=args.world)


if __name__ == "__main__":
    main()
