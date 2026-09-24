"""Tests for the notebook-output path scrubber (.scripts/ci/scrub_notebook_outputs.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRUBBER = Path(__file__).resolve().parents[1] / ".scripts" / "ci" / "scrub_notebook_outputs.py"
_spec = importlib.util.spec_from_file_location("scrub_notebook_outputs", _SCRUBBER)
scrub = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(scrub)


@pytest.mark.parametrize(
    "path",
    [
        "/localscratch/alice/ipykernel_571129/3300836858.py",
        "/tmp/ipykernel_42/1234.py",
        "/var/folders/xy/abc123/T/ipykernel_7/9.py",
        "/private/var/folders/xy/abc/T/ipykernel_7/9.py",
        "/ictstr01/groups/ml01/workspace/alice/.venv/lib/x.py",
        "/home/alice/env/lib/python3.12/site-packages/tqdm/auto.py",
        "/Users/alice/Library/Application Support/hatch/env/tqdm/auto.py",
        "/lustre/scratch/alice/run.py",
    ],
)
def test_scrubs_build_and_kernel_paths(path):
    """Every known build root and any ipykernel temp file is replaced."""
    text = f"{path}:21: UserWarning: something happened"
    out, count = scrub.scrub_paths(text)
    assert count >= 1
    assert path.split("/")[1] not in out  # the leading root segment is gone
    assert out.startswith("<path>")


@pytest.mark.parametrize(
    "text",
    [
        "see https://example.org/home/alice/page for details",
        "the module tmpfile helper (not a path)",
        "install from https://pypi.org/project/tqdm/",
    ],
)
def test_leaves_non_paths_untouched(text):
    """A URL or prose that merely contains a root word is not corrupted."""
    out, count = scrub.scrub_paths(text)
    assert count == 0
    assert out == text


def test_keeps_warning_text_only_scrubs_path():
    """A real warning keeps its message; only the absolute path is masked."""
    text = "/tmp/ipykernel_9/x.py:1: UserWarning: 2 of 467 features have no spread."
    out, _ = scrub.scrub_paths(text)
    assert out == "<path>:1: UserWarning: 2 of 467 features have no spread."


def test_idempotent_on_placeholder():
    """Running twice does not re-match the ``<path>`` it already wrote."""
    text = "/tmp/ipykernel_9/x.py:1: UserWarning: hi"
    once, _ = scrub.scrub_paths(text)
    twice, count = scrub.scrub_paths(once)
    assert count == 0
    assert twice == once


def test_drops_pure_noise_stderr():
    """A stderr block that is only tqdm/ipywidgets noise is removed."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": [
                            "/x/tqdm/auto.py:21: TqdmWarning: IProgress not found. "
                            "Please update jupyter and ipywidgets.\n",
                            "  from .autonotebook import tqdm as notebook_tqdm\n",
                        ],
                    }
                ],
            }
        ]
    }
    scrubbed, dropped = scrub.process_notebook(nb)
    assert dropped == 1
    assert nb["cells"][0]["outputs"] == []


def test_keeps_genuine_stderr_mentioning_ipywidgets():
    """A real diagnostic that merely names ipywidgets is not dropped as noise."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": "Widget rendering needs ipywidgets to be installed in this kernel.\n",
                    }
                ],
            }
        ]
    }
    _, dropped = scrub.process_notebook(nb)
    assert dropped == 0
    assert len(nb["cells"][0]["outputs"]) == 1


def test_scrubs_mixed_stderr_in_place():
    """A stderr that carries a real warning plus a path is kept and its path masked."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": "/localscratch/alice/ipykernel_1/z.py:1: UserWarning: keep me\n",
                    }
                ],
            }
        ]
    }
    scrubbed, dropped = scrub.process_notebook(nb)
    assert dropped == 0
    assert scrubbed == 1
    assert scrub._as_text(nb["cells"][0]["outputs"][0]["text"]) == (
        "<path>:1: UserWarning: keep me\n"
    )


def test_scrubs_error_traceback():
    """An error output's traceback frames have their paths masked."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": "ValueError",
                        "evalue": "boom",
                        "traceback": [
                            "Traceback (most recent call last):",
                            '  File "/home/alice/env/lib/python3.12/site-packages/x.py", line 3',
                        ],
                    }
                ],
            }
        ]
    }
    scrubbed, _ = scrub.process_notebook(nb)
    assert scrubbed == 1
    assert "/home/alice" not in "".join(nb["cells"][0]["outputs"][0]["traceback"])


def test_process_notebook_idempotent():
    """A second pass over an already-scrubbed notebook changes nothing."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": "/tmp/ipykernel_9/x.py:1: UserWarning: keep\n",
                    }
                ],
            }
        ]
    }
    scrub.process_notebook(nb)
    scrubbed, dropped = scrub.process_notebook(nb)
    assert (scrubbed, dropped) == (0, 0)
