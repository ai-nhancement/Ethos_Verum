#!/usr/bin/env python3
"""
api/server.py

Development server entrypoint for the Ethos REST API.

Usage:
    python -m api.server               (default: port 8000)
    python -m api.server --port 9000
    python -m api.server --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

_log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ethos REST API development server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=True,
                        help="Enable hot-reload on code changes (default: on)")
    parser.add_argument("--no-reload", dest="reload", action="store_false")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        _log.error("uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    _log.info("Starting Ethos API at http://%s:%d — docs at /docs", args.host, args.port)
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
