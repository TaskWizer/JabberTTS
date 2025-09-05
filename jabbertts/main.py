"""JabberTTS Main Application Module.

This module contains the main application logic and entry point for JabberTTS.
It sets up the FastAPI server, configures logging, and handles application lifecycle.
"""

import logging
import socket
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jabbertts import __version__
from jabbertts.config import get_settings
from jabbertts.api.routes import router as api_router


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


def main() -> None:
    """Main application entry point."""
    logger = None
    try:
        settings = get_settings()
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
