from __future__ import annotations

import argparse
import os

import uvicorn

from env import app as app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run("server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
