#!/usr/bin/env python3
"""Scrub build-environment leaks out of committed notebook outputs.

Notebooks under docs/ are committed with their executed outputs, and those
outputs are public. Captured warnings print absolute source paths, leaking the
build environment (username, cluster layout, virtualenv); the tqdm/ipywidgets
``IProgress not found`` warning is pure noise. For each ``.ipynb`` given, in
every code cell's outputs:

* In ``stream`` outputs and ``data["text/plain"]``, replace any absolute
  build-environment path with ``<path>``.
* Drop a ``stderr`` stream whose whole content is tqdm/ipywidgets noise or the
  "running over TCP" kernel notice; if it also carries real content, keep it
  and only scrub the paths.

Image data, execution counts, source and prose are left untouched. The pass is
idempotent. Default is FIX (rewrite changed files); ``--check`` writes nothing.
Either way the exit code is non-zero when a file changed, the pre-commit idiom.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PLACEHOLDER = "<path>"

# A run of path characters: slashes and the usual filename characters, but not
# whitespace, quotes, a colon (which ends the "<path>:<lineno>:" warning prefix)
# or angle brackets (so we never re-match a placeholder we already wrote).
_BODY = r"[^\s:'\"<>]"

# One absolute build-environment path. The roots are known build locations
# (``/home`` and ``/Users`` consume the user segment so an empty root cannot
# match); the final branch catches an interpreter path carrying a
# ``site-packages`` or ``.venv`` segment wherever it starts (the Library/
# "Application Support" paths break at the space, leaving this tail). The
# trailing lookbehind backtracks off sentence punctuation the path never owns.
_PATH = re.compile(
    rf"(?:/ictstr01|/lustre|/scratch|/home/{_BODY}+|/Users/{_BODY}+"
    rf"|/{_BODY}*(?:site-packages|\.venv)){_BODY}*(?<![.,;)])"
)

# Whole lines that are pure environment noise. Used only to decide whether a
# stderr block is droppable; kept blocks are never edited line-by-line.
_NOISE_LINES = [
    re.compile(r"(?m)^.*(?:TqdmWarning|IProgress not found|ipywidgets).*$"),
    re.compile(r"(?m)^\s*from \.autonotebook import tqdm as notebook_tqdm\s*$"),
    re.compile(r"(?m)^.*Kernel is running over TCP without encryption.*$"),
]


def scrub_paths(text: str) -> tuple[str, int]:
    """Replace absolute build-environment paths in ``text`` with ``<path>``."""
    return _PATH.subn(PLACEHOLDER, text)


def is_pure_noise(text: str) -> bool:
    """True if ``text`` holds noise lines and nothing else of substance."""
    residue = text
    for pattern in _NOISE_LINES:
        residue = pattern.sub("", residue)
    return residue.strip() == ""


def _as_text(field: str | list[str]) -> str:
    """A stream/text-plain field is a string or a list of lines; join it."""
    return field if isinstance(field, str) else "".join(field)


def _as_field(text: str, was_list: bool) -> str | list[str]:
    """Rebuild the field in its original shape, matching nbformat's line split."""
    if not was_list:
        return text
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def _scrub_field(container: dict, key: str) -> int:
    """Scrub paths in one string/list field in place; return replacement count."""
    value = container[key]
    text, count = scrub_paths(_as_text(value))
    if count:
        container[key] = _as_field(text, isinstance(value, list))
    return count


def process_notebook(nb: dict) -> tuple[int, int]:
    """Scrub and prune ``nb`` in place; return (paths scrubbed, blocks dropped)."""
    scrubbed = 0
    dropped = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        kept = []
        for output in outputs:
            kind = output.get("output_type")
            if kind == "stream":
                text = _as_text(output.get("text", ""))
                if output.get("name") == "stderr" and text.strip() and is_pure_noise(text):
                    dropped += 1
                    continue
                if "text" in output:
                    scrubbed += _scrub_field(output, "text")
            elif kind in ("execute_result", "display_data"):
                data = output.get("data", {})
                if "text/plain" in data:
                    scrubbed += _scrub_field(data, "text/plain")
            kept.append(output)
        if len(kept) != len(outputs):
            cell["outputs"] = kept
    return scrubbed, dropped


def _dumps(nb: dict) -> str:
    """Serialize like the repo's notebooks: indent=1, unicode kept, trailing \\n."""
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(args: argparse.Namespace) -> int:
    """Scrub each notebook; return 1 if any changed (or would change), else 0."""
    any_changed = False
    for name in args.paths:
        path = Path(name)
        if path.suffix != ".ipynb":
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        scrubbed, dropped = process_notebook(nb)
        if not (scrubbed or dropped):
            continue
        any_changed = True
        verb = "would scrub" if args.check else "scrubbed"
        drop_verb = "drop" if args.check else "dropped"
        print(f"{path}: {verb} {scrubbed} path(s), {drop_verb} {dropped} stderr block(s)")
        if not args.check:
            path.write_text(_dumps(nb), encoding="utf-8")
    return 1 if any_changed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="Notebook (.ipynb) paths to scrub.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would change and write nothing; exit non-zero if any file would change.",
    )
    raise SystemExit(main(parser.parse_args()))
