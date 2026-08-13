"""Convenience launcher: `python -m scripts.run_server`.

Equivalent to running `uvicorn src.api:app --host ... --port ...` by hand,
but picks up API_HOST/API_PORT from .env automatically so there's one less
thing to get wrong when starting the server.
"""

from __future__ import annotations

import uvicorn

from src.config import settings


def main() -> None:
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()