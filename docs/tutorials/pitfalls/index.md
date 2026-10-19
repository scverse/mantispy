# Pitfalls

A trustworthy backend teaches its own failure modes. Image-based profiling has a handful of mistakes that produce
a confident, plausible-looking number from data that cannot support it, and none of them raise an error. This tier
collects them: what the symptom looks like, and how mantispy lets you detect it.

```{toctree}
:hidden:
:maxdepth: 1

pseudoreplication
```

- **[Pseudoreplication](pseudoreplication.ipynb).** *Symptom:* a treatment looks highly significant because every cell in a well is counted
  as an independent replicate, inflating the sample size a hundredfold. *Fix:* aggregate to the well (or plate)
  first with `tl.aggregate`, and treat wells, not cells, as the unit of replication.

The unit-of-replication and p-value-floor pitfalls are worked through in full in
[Which measurements moved](../phenotypes/which_features_moved.ipynb).
