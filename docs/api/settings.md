# Settings

```{eval-rst}
.. currentmodule:: mantispy
```

Every setting is also read from a `MANTISPY_` environment variable, and `with mt.settings.override(cache_dir=...):` changes one for a block.

mantispy logs through the `scverse.mantispy` logger, which shares its level with the other scverse packages.
Set it for all of them with `scverse_misc.logging.config.verbosity = "info"`, or for mantispy alone with `logging.getLogger("scverse.mantispy").setLevel("INFO")`.

```{eval-rst}
.. autosummary::
    :toctree: generated

    settings
    settings.override
    settings.reset
```
