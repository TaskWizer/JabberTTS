"""JabberTTS Main Application Module.

This module contains the main application logic and entry point for JabberTTS.
It sets up the FastAPI server, configures logging, and handles application lifecycle.
"""

import argparse
import logging
import socket
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jabbertts import __version__
from jabbertts.config import get_settings
from jabbertts.api.routes import router as api_router
from jabbertts.dashboard.routes import router as dashboard_router
from jabbertts.voice_cloning.routes import router as voice_cloning_router


def check_port_availability(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    Args:
        host: Host address to check
        port: Port number to check

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> Optional[int]:
    """Find an available port starting from the given port.

    Args:
        host: Host address to check
        start_port: Starting port number
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number or None if no port found
    """
    for port in range(start_port, start_port + max_attempts):
        if check_port_availability(host, port):
            return port
    return None


def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title="JabberTTS API",
        description="Fast, efficient Text-to-Speech API with OpenAI compatibility",
        version=__version__,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router, prefix="/v1")

    # Include dashboard routes
    app.include_router(dashboard_router, prefix="/dashboard")

    # Include voice cloning studio routes
    app.include_router(voice_cloning_router, prefix="/studio")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        return {
            "status": "healthy",
            "version": __version__,
            "service": "jabbertts"
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with basic information."""
        return {
            "service": "JabberTTS",
            "version": __version__,
            "description": "Fast, efficient Text-to-Speech API with OpenAI compatibility",
            "docs": "/docs" if settings.enable_docs else "Documentation disabled",
            "health": "/health"
        }
    
    return app


def parse_cli_args() -> Dict[str, Any]:
    """Parse command-line arguments.

    Returns:
        Dictionary of CLI arguments for configuration override
    """
    parser = argparse.ArgumentParser(
        description="JabberTTS - Fast, efficient Text-to-Speech API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Start with default settings
  %(prog)s --port 8001                        # Use custom port
  %(prog)s --host 127.0.0.1 --port 8080      # Localhost only on port 8080
  %(prog)s --config ./custom/config          # Use custom config directory
  %(prog)s --audio-quality high              # Use high audio quality
  %(prog)s --debug                           # Enable debug mode

Configuration precedence (highest to lowest):
  1. Command-line arguments
  2. override.json file
  3. settings.json file
  4. Environment variables (JABBERTTS_*)
  5. Default values
        """
    )

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        help="Server host address (default: 0.0.0.0 for network access, 127.0.0.1 for localhost only)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes (default: 1)"
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./config",
        help="Configuration directory path (default: ./config)"
    )

    # Audio settings
    parser.add_argument(
        "--audio-quality",
        choices=["low", "standard", "high", "ultra"],
        help="Audio quality preset (default: standard)"
    )

    # Development options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    # Version
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"JabberTTS {__version__}"
    )

    args = parser.parse_args()

    # Convert to dictionary, filtering out None values
    cli_config = {}
    if args.host is not None:
        cli_config["host"] = args.host
    if args.port is not None:
        cli_config["port"] = args.port
    if args.workers is not None:
        cli_config["workers"] = args.workers
    if args.audio_quality is not None:
        cli_config["audio_quality"] = args.audio_quality
    if args.debug:
        cli_config["debug"] = True
    if args.reload:
        cli_config["reload"] = True
    if args.log_level is not None:
        cli_config["log_level"] = args.log_level

    return cli_config, args.config


def main() -> None:
    """Main application entry point."""
    logger = None
    try:
        # Parse CLI arguments
        cli_config, config_dir = parse_cli_args()

        # Get settings with CLI overrides
        settings = get_settings(config_dir=config_dir, **cli_config)
        setup_logging(settings.log_level)

        logger = logging.getLogger(__name__)
        logger.info(f"Starting JabberTTS v{__version__}")
        logger.info(f"Configuration: {settings.model_dump()}")

        # Check if the configured port is available
        if not check_port_availability(settings.host, settings.port):
            logger.warning(f"Port {settings.port} is already in use on {settings.host}")

            # Try to find an alternative port
            alternative_port = find_available_port(settings.host, settings.port + 1)
            if alternative_port:
                logger.info(f"Found alternative port: {alternative_port}")
                logger.info(f"You can set JABBERTTS_PORT={alternative_port} to use this port permanently")

                # Check if auto_port is enabled or if we're in a non-interactive environment
                if settings.auto_port or not sys.stdin.isatty():
                    settings.port = alternative_port
                    logger.info(f"Automatically using alternative port: {alternative_port}")
                else:
                    # Ask user if they want to use the alternative port
                    try:
                        response = input(f"Port {settings.port} is in use. Use port {alternative_port} instead? (y/N): ")
                        if response.lower() in ['y', 'yes']:
                            settings.port = alternative_port
                            logger.info(f"Using alternative port: {alternative_port}")
                        else:
                            logger.error("Cannot start server: port is not available")
                            logger.error("Set JABBERTTS_AUTO_PORT=true to automatically use alternative ports")
                            sys.exit(1)
                    except (EOFError, KeyboardInterrupt):
                        logger.error("Cannot start server: port is not available")
                        logger.error("Set JABBERTTS_AUTO_PORT=true to automatically use alternative ports")
                        sys.exit(1)
            else:
                logger.error(f"No available ports found starting from {settings.port + 1}")
                logger.error("Please specify a different port using JABBERTTS_PORT environment variable")
                sys.exit(1)

        # Create the FastAPI app
        app = create_app()

        # Run the server with socket reuse enabled
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
            log_level=settings.log_level.lower(),
            access_log=settings.access_log,
            reload=settings.reload,
            # Enable socket reuse to handle port binding issues
            server_header=False,
            date_header=False,
        )

    except KeyboardInterrupt:
        if logger:
            logger.info("Shutting down JabberTTS...")
    except Exception as e:
        if logger:
            logger.error(f"Failed to start JabberTTS: {e}")
        else:
            print(f"Failed to start JabberTTS: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
