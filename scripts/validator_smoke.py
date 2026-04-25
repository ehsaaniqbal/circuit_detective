from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from socket import socket
from urllib.error import URLError
from urllib.request import urlopen

from circuit_detective import CircuitDetectiveAction, CircuitDetectiveEnv


ROOT = Path(__file__).resolve().parents[1]


def reserve_port() -> int:
    with socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(base_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(1.0)
    raise TimeoutError("Server did not become healthy in time.")


def main() -> None:
    port = reserve_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "circuit_detective.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_server(base_url)
        with CircuitDetectiveEnv(base_url=base_url).sync() as client:
            reset = client.reset()
            assert reset.observation.scenario_id == "l1_induction_attn_only_2l"

            step = client.step(CircuitDetectiveAction(tool_name="list_tools"))
            assert step.observation.done is False

            state = client.state()
            assert state.step_count >= 1

        print("validator_smoke: ok")
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
