# Choosing a readout

Several functions score how well a screen reproduces and how active its perturbations are.
They answer different questions, so the one to report depends on what you want to claim.

| Readout | Question it answers | Report it when |
| --- | --- | --- |
| `tl.map(mode="activity")` | Is this perturbation distinguishable from the negative controls on its own plate? | You want to know which perturbations do anything. This is the default first readout for a screen with controls. |
| `tl.map(mode="replicability")` | Do a perturbation's replicates retrieve each other against the other perturbations on a plate, controls left out? | You want replicate reproducibility within a plate. |
| `tl.map(mode="cross_plate")` | Do a perturbation's replicates on other plates retrieve each other against all other profiles? | Replicates sit on different plates or batches, and you need reproducible biology rather than plate effects. |
| `tl.map(mode="consistency")` | Do perturbations that share an annotation, such as a mechanism or a target, look more alike than those that do not? | You have consensus profiles of active perturbations and a mechanism or gene-set annotation to test. |
| `tl.percent_replicating` | Is the median replicate correlation above a non-replicate null? | You must compare with published numbers. It thresholds rather than ranks and depends on the number of replicates, so prefer mAP otherwise. |
| `tl.hit_calling` | Is this perturbation's profile far from the controls, with a calibrated call per perturbation? | You need a list of hits rather than a score for every perturbation. |
| `tl.edistance` | How far apart are two whole distributions of profiles? | You want a distance between groups, or between each group and the controls, that uses the full distribution and not only its centre. |

## Which to report

Start with `activity` to find the perturbations that differ from the controls.
Report `cross_plate` next to it whenever replicates are spread over plates, because a signal that does not survive a plate change is not reproducible.
Add `consistency` on the active perturbations to ask whether they group by mechanism or target.
Use `hit_calling` when the deliverable is a hit list, and `edistance` when the deliverable is a distance.
Report `percent_replicating` only for comparison with earlier work.

`tl.map` needs the optional copairs extra.
See {doc}`screen_quality` for the readouts applied to a screen, and the {func}`~mantispy.tl.map` reference for the pair definitions behind each mode.
