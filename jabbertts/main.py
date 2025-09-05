"""JabberTTS Main Application Module.

This module contains the main application logic and entry point for JabberTTS.
It sets up the FastAPI server, configures logging, and handles application lifecycle.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jabbertts import __version__
from jabbertts.config import get_settings
from jabbertts.api.routes import router as api_router


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
    try:
        settings = get_settings()
        setup_logging(settings.log_level)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting JabberTTS v{__version__}")
        logger.info(f"Configuration: {settings.model_dump()}")
        
        # Create the FastAPI app
        app = create_app()
        
        # Run the server
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
            log_level=settings.log_level.lower(),
            access_log=settings.access_log,
            reload=settings.reload,
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down JabberTTS...")
    except Exception as e:
        logger.error(f"Failed to start JabberTTS: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
