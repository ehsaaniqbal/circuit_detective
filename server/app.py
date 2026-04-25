"""FastAPI application for Circuit Detective."""

from __future__ import annotations

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError("Install project dependencies with `uv sync`.") from exc

try:
    from ..models import CircuitDetectiveAction, CircuitDetectiveObservation
    from .circuit_detective_environment import CircuitDetectiveEnvironment
except ImportError:
    from models import CircuitDetectiveAction, CircuitDetectiveObservation
    from server.circuit_detective_environment import CircuitDetectiveEnvironment


app = create_app(
    CircuitDetectiveEnvironment,
    CircuitDetectiveAction,
    CircuitDetectiveObservation,
    env_name="circuit_detective",
    max_concurrent_envs=16,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    resolved_host = os.getenv("HOST", host)
    resolved_port = int(os.getenv("PORT", str(port)))
    uvicorn.run(app, host=resolved_host, port=resolved_port)


if __name__ == "__main__":
    main()
