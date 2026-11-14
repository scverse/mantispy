# tl.map and copairs

{func}`~mantispy.tl.map` computes the mean average precision (mAP) that [copairs](https://github.com/cytomining/copairs) defines, with its own compiled code instead of a call to copairs.
With its default settings it gives the same scores, p-values and q-values as copairs 0.6.0 for the same seed, so results published with copairs reproduce.
The page lists how the copairs arguments map onto `tl.map`, what each null assumes, and which settings keep that agreement.

## Arguments

`tl.map` takes the arguments of copairs' `average_precision` and `mean_average_precision` in one call.

  | copairs                                                   | `tl.map`                                                                                                                            |
  | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
  | `pos_sameby`, `pos_diffby`, `neg_sameby`, `neg_diffby`    | The same names, or `mode=` for a preset that fills all four.                                                                        |
  | `copairs.map.multilabel.average_precision`                | A list-valued `annotation_key`, or a delimited string split with `label_sep=`.                                                      |
  | `distance`, `null_size`, `threshold`, `seed`, `null`      | The same names. The default `null` differs, as the next section explains.                                                           |
  | `max_workers`                                             | numba's threads, one per core by default. Cap them with `numba.set_num_threads`.                                                    |
  | `batch_size`                                              | Not needed, because `tl.map` never stores the pairs, so memory does not grow with their number.                                     |
  | `cache_dir`                                               | Not needed. Each null is drawn from a seed of its own, so a p-value does not depend on earlier calls or a cache on disk.           |
  | `pos_pairs`, `pairs`                                      | Not needed, because `tl.map` scores the pairs itself and knows them for every null.                                                 |
  | `mean_average_precision_hierarchical`                     | No equivalent.                                                                                                                      |

## Nulls

A null is the distribution the group mAP would have if the group's label meant nothing, and the p-value is the share of null draws that reach the observed mAP.
copairs and `tl.map` offer the same three nulls.

- `"copied"` compares each member's average precision with random rankings that have the member's numbers of positives and pairs, and averages over the members. Members with the same numbers share one random draw, which widens the null, so the test is conservative for groups with several replicates. It is the model of Kalinin et al. (2025) and the default of copairs before 0.6.
- `"joint"` draws the members' average precisions together, so a positive pair between two members is one random rank in both of their lists. It can give p-values that are too small for groups with many replicates, which copairs warns about too.
- `"permutation"` moves the group's label at random among the profiles its members are ranked with, and recomputes the mAP from the observed similarities. It is the slowest of the three.

copairs 0.6 made `"permutation"` its default, and `tl.map` keeps `"copied"`.
Pass `null="permutation"` to match a copairs 0.6 run that used its default.

## Reproducing copairs numbers

The settings fall into two levels of agreement with copairs.

- `null="copied"` with `null_sampler="exact"`, and `null="joint"`, replay copairs' random streams, so every p-value equals copairs' for the same seed, to the last digit. Use them for numbers you report.
- `null_sampler="floyd"` draws the copied null from the same distribution in less time on large screens, but with other random draws, so its p-values differ from copairs' by the random error of the draws.

`null="permutation"` is a separate implementation of copairs' permutation null, and its random draws are not copairs' draws.
It was checked statistically instead of draw by draw.
On the negative controls of JUMP-CRISPR under made-up labels, where no group is real, its p-values are uniform.
On the phenotypic activity of JUMP-CRISPR over 12 seeds, it calls 7,589 groups on average at a false discovery rate of 5%, and copairs calls 7,589, with a spread of about 2 calls across seeds for both.
