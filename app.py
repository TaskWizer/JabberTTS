#!/usr/bin/env python3
"""JabberTTS Application Entry Point.

This is the main entry point for the JabberTTS application.

Usage:
    python app.py [options]

Examples:
    python app.py --port 8001
    python app.py --host 127.0.0.1 --audio-quality high
    python app.py --config ./custom/config --debug
    python app.py --help

For full usage information, run: python app.py --help
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from jabbertts.main import main

    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Error importing JabberTTS: {e}")
    print("Please ensure all dependencies are installed with: uv sync")
    sys.exit(1)
except KeyboardInterrupt:
    print("\nShutting down JabberTTS...")
    sys.exit(0)
except Exception as e:
    print(f"Failed to start JabberTTS: {e}")
    sys.exit(1)