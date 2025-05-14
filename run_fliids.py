import argparse
import os
import sys
import subprocess
from utils import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="baseline", choices=["baseline", "dem", "full"]
    )
    parser.add_argument("--clients", type=int, default=3)
    args = parser.parse_args()

    # Apply global config
    config.apply_config(args.mode)
    print(f"Running FL-IIDS in mode: {args.mode.upper()}")

    # Start server
    print("Starting server...")
    server_proc = subprocess.Popen([sys.executable, "server.py"])

    # Start clients
    for i in range(args.clients):
        env = os.environ.copy()
        env["CLIENT_ID"] = f"client{i}"
        subprocess.Popen([sys.executable, "client.py"], env=env)

    # Optionally: wait for server_proc to end
    server_proc.wait()
    print("FL training finished.")


if __name__ == "__main__":
    main()
