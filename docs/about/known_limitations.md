# Known limitations

mantispy is in alpha. These are the gaps to know about before you build on it.

- **Case studies are not re-executed in CI.** The tutorials run on every change, but the case-study notebooks ship with stored outputs and are only re-run by hand, so their outputs can lag behind the code.
- **One perturbation per object, by design.** Each row is a member of a single perturbation. A cell carrying more than one guide is a QC artifact that is expected to be filtered out upstream, in your own pipeline, before stamping, so multi-membership is out of scope rather than unimplemented. In pooled screens the guide is the perturbation identity (recorded in `uns["mantispy"]["perturbation"]`) and is aggregated to genes downstream; see {doc}`../tutorials/genetic_screens`.
- **Performance is not tuned for full JUMP scale.** The functions are correct on large screens, but some use memory or time that grows quickly with the number of profiles. One easy lever is the PCA that feeds {func}`~mantispy.pp.tvn` and {func}`~mantispy.pp.harmony`: scanpy's default solver decomposes the whole cells-by-features matrix in double precision, whereas on a screen with far more cells than features `sc.pp.pca(adata, svd_solver="covariance_eigh")` eigendecomposes the small features-by-features covariance instead. It returns the same components with a fraction of the memory and time, and is exact rather than fitted on a subsample.
- **SpatialData is read-only.** Images and segmentations are read through `io.read_plate`, and nothing is written back to the SpatialData object.
