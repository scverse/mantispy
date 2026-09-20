#!/usr/bin/env python3
"""Download datasets to populate CI cache.

This script downloads every dataset the tutorials need.
The loaders cache to mantispy.settings.cache_dir.
"""

from __future__ import annotations

import argparse

_CNT = 0  # increment this when you want to rebuild the CI cache


def main(args: argparse.Namespace) -> None:
    """Call each loader once, so its files and anything it assembles from them land in the cache."""
    import mantispy as mt

    loaders = {
        "bbbc021": mt.ds.bbbc021,
        "rohban": mt.ds.rohban,
        "pki": mt.ds.pki,
        # The default, one plate from each source, is what the tutorials read. All 141 would be 9.4 GB,
        # against the 10 GB GitHub gives a whole repository for its caches.
        "jump_target2": mt.ds.jump_target2,
        "jump_cells": mt.ds.jump_cells,
        "jump_plate": mt.ds.jump_plate,
        "oasis_pilot": mt.ds.oasis_pilot,
    }
    if args.dry_run:
        print(f"Cache: {mt.settings.cache_dir}\nWould download: {', '.join(loaders)}")
        return

    for load in loaders.values():
        load()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets to populate CI cache.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download, just print what would be downloaded.",
    )

    main(parser.parse_args())
