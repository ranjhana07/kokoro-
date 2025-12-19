#!/usr/bin/env python3
import os
import sys
from importlib import import_module


def _project_root_from_here() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_host_port():
    host = os.environ.get("HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("PORT", "7860"))
    except ValueError:
        port = 7860
    return host, port


def main():
    # Ensure repository root is importable so we can import `web_ui`
    repo_root = _project_root_from_here()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        web_ui = import_module("web_ui")
    except Exception as e:
        print("Could not import 'web_ui' from repo root.")
        print(f"Error: {e}")
        print("Make sure a file named 'web_ui.py' exists at the repository root.")
        sys.exit(1)

    host, port = _get_host_port()

    # Provide optional hints for chunking/latency to downstream code
    os.environ.setdefault("KOKORO_LOW_LATENCY", os.environ.get("KOKORO_LOW_LATENCY", "1"))
    # Example: allow overriding chunk size via env; your web_ui can read this
    os.environ.setdefault("KOKORO_CHUNK_SIZE", os.environ.get("KOKORO_CHUNK_SIZE", "1000"))

    # 0) Prefer explicit entrypoints in web_ui to preserve custom logic
    for fname in ("main", "run", "serve", "start"):
        entry = getattr(web_ui, fname, None)
        if callable(entry):
            print(f"Starting via web_ui.{fname}() on http://{host}:{port} ...")
            try:
                # Try passing host/port if supported
                entry(host=host, port=port)
            except TypeError:
                # Fallback: call without args
                entry()
            return

    # 1) FastAPI/Starlette style: `app = FastAPI()`
    app = getattr(web_ui, "app", None) or getattr(web_ui, "application", None)
    if app is not None and hasattr(app, "router"):
        try:
            import uvicorn  # type: ignore
        except ImportError:
            print("uvicorn is required to run a FastAPI/Starlette app.")
            print("Install it with: pip install uvicorn[standard]")
            sys.exit(1)

        print(f"Starting FastAPI/Starlette app on http://{host}:{port} ...")
        uvicorn.run(app, host=host, port=port)
        return

    # 2) Gradio style: object with `.launch()` such as `demo`, `ui`, `iface`
    for candidate in ("demo", "ui", "iface", "interface", "app"):
        ui = getattr(web_ui, candidate, None)
        if ui is not None and hasattr(ui, "launch"):
            print(f"Starting Gradio app on http://{host}:{port} ...")
            ui.launch(server_name=host, server_port=port, show_error=True)
            return

    # 3) Flask style: `flask_app = Flask(__name__)` or `app = Flask(__name__)`
    for candidate in ("flask_app", "app", "application"):
        flask_app = getattr(web_ui, candidate, None)
        if flask_app is not None and hasattr(flask_app, "run") and hasattr(flask_app, "route"):
            print(f"Starting Flask app on http://{host}:{port} ...")
            flask_app.run(host=host, port=port)
            return

    print("No known server object found in 'web_ui.py'.")
    print("Expected one of: FastAPI 'app', Gradio 'ui/demo/iface', or Flask 'flask_app'.")
    sys.exit(2)


if __name__ == "__main__":
    main()
