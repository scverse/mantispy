# Pitfalls

A trustworthy backend teaches its own failure modes. Image-based profiling has a handful of mistakes that produce
a confident, plausible-looking number from data that cannot support it, and none of them raise an error. This tier
collects them: what the symptom looks like, and how mantispy lets you detect it.

The pages that will live here:

- **Pseudoreplication.** *Symptom:* a treatment looks highly significant because every cell in a well is counted
  as an independent replicate, inflating the sample size a hundredfold. *Fix:* aggregate to the well (or plate)
  first with `tl.aggregate`, and treat wells, not cells, as the unit of replication.
- **Wrong normalization for compounds.** *Symptom:* hits and effect sizes shift when a plate's staining drifts,
  because features were normalized without anchoring to a negative control. *Fix:* normalize each plate against
  its own DMSO wells (`reference="negcon"`, `by="Metadata_Plate"`), so a treated well is read relative to the
  controls it shares a plate with.
- **Reading a dose curve that is not there.** *Symptom:* a smooth-looking concentration response fitted to two or
  three doses with wide replicate scatter, reported as a potency. *Fix:* check the replicate spread and the number
  of doses before fitting, and let the fit report its own uncertainty rather than a single EC50.
- **A batch correction that makes retrieval worse.** *Symptom:* a correction removes visible plate structure but
  lowers mechanism retrieval, because it also removed biology aligned with the batch. *Fix:* score the correction
  with `metrics.evaluate_correction` against the uncorrected run, and keep it only when retrieval improves.
- **The p-value floor of a rank test.** *Symptom:* no group passes multiple-testing correction, or every group
  shares an identical smallest p-value, because a permutation or rank null cannot resolve below `1 / (n + 1)`.
  *Fix:* raise the number of permutations (or `null_size`) until the floor sits below the corrected threshold you
  need.
- **Clustering that follows the perturbation.** *Symptom:* clusters recover the experimental design (which plate,
  which batch) rather than biology, and are then interpreted as phenotypes. *Fix:* check whether a cluster maps
  onto a technical covariate before reading it as a cell state.

The unit-of-replication and p-value-floor pitfalls are worked through in full in
[Which measurements moved](../phenotypes/which_features_moved.ipynb).
