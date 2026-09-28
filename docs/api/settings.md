# Settings

```{eval-rst}
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    settings
    settings.override
    settings.reset
```

Both settings, `verbosity` and `cache_dir`, are also read from `MANTISPY_VERBOSITY` and `MANTISPY_CACHE_DIR`, and `with mt.settings.override(verbosity=2):` changes one for a block.

## Seeing what mantispy is doing

mantispy logs what it drops, skips and shrinks through the standard library's `logging`.
Warnings print by default; raise the verbosity to see the rest:

```python
import mantispy as mt

mt.settings.verbosity = 2  # 0 errors, 1 warnings (default), 2 info, 3 debug
```

Set it to 2 when you first run a new screen.
Several functions discard features or wells, and they log it at that level.
