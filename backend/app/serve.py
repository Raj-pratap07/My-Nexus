"""
Local launcher:  python -m app.serve

Binds to the loopback interface only. Settings validation refuses any
non-loopback NEXUS_HOST, so the development-identity API cannot be exposed
to a network by configuration. (Running uvicorn directly with --host
0.0.0.0 bypasses this launcher; don't.)
"""
from __future__ import annotations

import logging
import sys

import uvicorn

from app.core.config import Settings
from app.core.errors import StartupError


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        settings = Settings.from_env()
    except StartupError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    uvicorn.run(
        "app.main:create_app_from_env",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
