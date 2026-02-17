"""Entry point for running the JuniperData API with uvicorn."""

import argparse
import sys


def main() -> int:
    """Run the JuniperData API server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install 'juniper-data[api]'")
        return 1

    from juniper_data.api.settings import Settings

    parser = argparse.ArgumentParser(
        description="Run the JuniperData API server",
        prog="python -m juniper_data",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from settings, which default to 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from settings or 8100)",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=None,
        help="Path to dataset storage directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["TRACE", "VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "FATAL"],
        default=None,
        help="Logging level",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    settings = Settings()

    host = args.host if args.host is not None else settings.host
    port = args.port if args.port is not None else settings.port
    log_level_source = args.log_level if args.log_level is not None else settings.log_level
    log_level = log_level_source.lower()

    if args.storage_path is not None:
        import os

        os.environ["JUNIPER_DATA_STORAGE_PATH"] = args.storage_path

    uvicorn.run(
        "juniper_data.api.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=args.reload,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
