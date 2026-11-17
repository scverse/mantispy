#!/usr/bin/env python3
"""Download datasets to populate CI cache.

This script downloads every dataset the tutorials need.
The loaders cache to mantispy.settings.cache_dir.
"""

from __future__ import annotations

import argparse

_CNT = 3  # increment this when you want to rebuild the CI cache

# The plates docs/tutorials/multisite/cross_laboratory.ipynb loads by name for its cross-source hierarchy.
_TARGET2_HIERARCHY = [
    "JCPQC051", "JCPQC052", "JCPQC053", "JCPQC054",
    "BR00121438", "BR00121439", "BR00126113", "BR00126114",
    "110000294936", "110000296682", "110000296339", "110000296356",
]  # fmt: skip


def main(args: argparse.Namespace) -> None:
    """Call each loader once, so its files and anything it assembles from them land in the cache."""
    import pooch

    import mantispy as mt

    def jump_lite_all() -> None:
        """Every feature set, because tutorial 12 compares them and each is a separate file."""
        for model in mt.ds.JUMP_LITE_MODELS:
            mt.ds.jump_lite(model=model)
        mt.ds.jump_lite_targets()

    loaders = {
        "bbbc021": mt.ds.bbbc021,
        "rohban": mt.ds.rohban,
        # The staged rohban variants, so tutorials and tests that load them hit the cache (<7 MB each).
        "rohban_selected": lambda: mt.ds.rohban(feature_selected=True),
        "rohban_gene": lambda: mt.ds.rohban(aggregated=True),
        "rohban_gene_selected": lambda: mt.ds.rohban(aggregated=True, feature_selected=True),
        "pki": mt.ds.pki,
        "aconcagua_cells": mt.ds.aconcagua_cells,
        # Brieflow's published interphase gene table, which docs/tutorials/genetics/coming_from_brieflow.ipynb compares
        # against; mirrored from s3://ops-explorer-public/aconcagua/aconcagua/visualizations/interphase/aggregated_data.h5ad.
        "aconcagua_brieflow_interphase": lambda: pooch.retrieve(
            "https://exampledata.scverse.org/mantispy/aconcagua/aconcagua_brieflow_interphase.h5ad",
            known_hash="sha256:f55756939edc8ddf108092d88c4efe248765b4db4ec39322d4b4a988e04af929",
            fname="aconcagua_brieflow_interphase.h5ad",
            path=mt.settings.cache_dir,
        ),
        # These two come to ~1.2 GB; all 141 plates would be 9.4 GB, over GitHub's 10 GB cache.
        "jump_target2": mt.ds.jump_target2,
        "jump_target2_hierarchy": lambda: mt.ds.jump_target2(plates=_TARGET2_HIERARCHY),
        "jump_cells": mt.ds.jump_cells,
        "jump_plate": mt.ds.jump_plate,
        "jump_export": mt.ds.jump_export,
        "jump_lite": jump_lite_all,
        "oasis_pilot": mt.ds.oasis_pilot,
        "jump_crispr": mt.ds.jump_crispr,
        "corum": mt.ds.corum,
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
