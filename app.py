#!/usr/bin/env python3
"""JabberTTS Application Entry Point.

This is the main entry point for the JabberTTS application.
Run with: uv run python app.py
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