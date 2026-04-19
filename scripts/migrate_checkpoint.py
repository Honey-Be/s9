#!/usr/bin/env python3
"""Convenience entry point for checkpoint migration.

Usage::

    python scripts/migrate_checkpoint.py --in old.pt --out new.pt [--family auto] [--direction to_zoh]
"""
from s9.migration.cli import main

if __name__ == "__main__":
    main()
