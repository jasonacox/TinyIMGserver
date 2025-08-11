#!/usr/bin/env python3
"""
Version information script for TinyIMGserver.

This script displays the current version and related information.
"""

import sys
import os

# Add the current directory to the path to import from app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.__version__ import __version__, __title__, __description__, __author__, __license__

def main():
    """Display version information."""
    print(f"{__title__} v{__version__}")
    print(f"Description: {__description__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")

if __name__ == "__main__":
    main()
