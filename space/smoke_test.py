"""Exercise the Space's actual launch path and all callbacks on loopback only."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from gradio_client import Client


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, GRADIO_SERVER_NAME="127.0.0.1",
               GRADIO_SERVER_PORT=str(port), GRADIO_ANALYTICS_ENABLED="False",
               PYTHONUTF8="1")
    with tempfile.TemporaryDirectory() as scratch:
        log_path = Path(scratch) / "app.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("app.py"))],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 90
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Space exited before startup")
                    try:
                        with urllib.request.urlopen(url + "/config", timeout=2) as response:
                            config = json.load(response)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Space did not start within 90 seconds")
                        time.sleep(0.25)
                tabs = [c for c in config["components"] if c["type"] == "tabitem"]
                assert len(tabs) == 5, "Expected all five application tabs"
                assert ".main-header" in config["css"], "Launch-time CSS was not preserved"
                client = Client(url, verbose=False)
                table, plot = client.predict(27, api_name="/run_live_benchmark")
                assert "Avg shortest path" in table and plot["type"] == "plotly"
                summary, recall, spatial = client.predict(50, api_name="/run_index_benchmark")
                assert "FCC recall" in summary and recall["type"] == spatial["type"] == "plotly"
                for distribution in ("Uniform", "Random", "Corpus"):
                    summary, spectral, weights = client.predict(
                        27, distribution, api_name="/run_weighted_benchmark"
                    )
                    assert "Fiedler ratio" in summary
                    assert spectral["type"] == weights["type"] == "plotly"
                print("PASS: startup, five tabs, custom CSS, and all five callback cases")
            except Exception:
                log.flush()
                print(log_path.read_text(encoding="utf-8"), file=sys.stderr)
                raise
            finally:
                if os.name == "nt" and process.poll() is None:
                    # Windows venv launchers spawn a second Python process.
                    # Stop only this owned process tree, including that child.
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=True, stdout=subprocess.DEVNULL, timeout=15,
                    )
                else:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)


if __name__ == "__main__":
    main()
