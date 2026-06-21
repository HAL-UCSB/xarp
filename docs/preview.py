"""Build and preview the XARP Sphinx documentation."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import subprocess
import sys
import webbrowser


_BOOTSTRAP_ENV = "XARP_DOCS_BOOTSTRAPPED"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and locally preview the XARP API documentation."
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="build the HTML documentation without starting a server",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not open the documentation in a browser",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="server bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="server port (default: 8000)",
    )
    return parser


def _find_uv() -> str | None:
    configured = os.environ.get("UV")
    candidates = [
        configured,
        shutil.which("uv"),
        str(Path(sys.executable).with_name("uv.exe")),
        str(Path(sys.executable).with_name("uv")),
    ]
    return next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)


def _bootstrap_docs_extra(arguments: list[str]) -> int | None:
    """Relaunch with the documentation extra when Sphinx is unavailable."""
    if find_spec("sphinx") is not None:
        return None

    if os.environ.get(_BOOTSTRAP_ENV):
        print(
            "Sphinx is unavailable after loading the docs extra.",
            file=sys.stderr,
        )
        return 1

    docs_dir = Path(__file__).resolve().parent
    project_dir = docs_dir.parent
    if not (project_dir / "pyproject.toml").is_file():
        print(
            "The docs command requires Sphinx. Install this package with "
            "the docs extra: pip install 'xarp[docs]'.",
            file=sys.stderr,
        )
        return 1

    uv = _find_uv()
    if uv is None:
        print(
            "uv was not found. Install uv or install the xarp[docs] extra manually.",
            file=sys.stderr,
        )
        return 1

    environment = os.environ.copy()
    environment[_BOOTSTRAP_ENV] = "1"
    command = [
        uv,
        "run",
        "--project",
        str(project_dir),
        "--extra",
        "docs",
        "docs",
        *arguments,
    ]
    return subprocess.call(command, env=environment)


def _browser_url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{port}/"


def main() -> int:
    """Build the documentation and optionally serve it locally."""
    arguments = sys.argv[1:]
    bootstrap_result = _bootstrap_docs_extra(arguments)
    if bootstrap_result is not None:
        return bootstrap_result

    options = _parser().parse_args(arguments)
    docs_dir = Path(__file__).resolve().parent
    output_dir = docs_dir / "_build" / "html"

    from sphinx.cmd.build import main as sphinx_build

    build_result = sphinx_build(
        [
            "-W",
            "--keep-going",
            "-E",
            "-b",
            "html",
            str(docs_dir),
            str(output_dir),
        ]
    )
    if build_result:
        return build_result

    index = output_dir / "index.html"
    if options.build_only:
        print(f"Documentation built at {index}")
        return 0

    handler = partial(SimpleHTTPRequestHandler, directory=str(output_dir))
    try:
        server = ThreadingHTTPServer((options.host, options.port), handler)
    except OSError as error:
        print(
            f"Could not serve documentation on {options.host}:{options.port}: {error}",
            file=sys.stderr,
        )
        return 1

    url = _browser_url(options.host, options.port)
    print(f"Serving documentation at {url} (press Ctrl+C to stop)")
    if not options.no_open and not webbrowser.open(url):
        print(f"Could not open a browser automatically. Open {url} manually.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping documentation server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
