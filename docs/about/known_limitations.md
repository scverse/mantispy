# Known limitations

mantispy is in alpha. These are the gaps to know about before you build on it.

- **Case studies are not re-executed in CI.** The tutorials run on every change, but the case-study notebooks ship with stored outputs and are only re-run by hand, so their outputs can lag behind the code.
- **One object, one perturbation.** A cell carrying two barcodes or guides cannot yet be represented as a member of both perturbations.
- **Learned-embedding provenance is optional.** `mt.io.stamp(..., provenance={"model": ..., "version": ..., "pooling": ...})` records the model, its version and the pooling in `uns["mantispy"]["embedding"]`, but nothing requires it, and `validate` only warns when `model` is missing.
- **Performance is not tuned for full JUMP scale.** The functions are correct on large screens, but some use memory or time that grows quickly with the number of profiles.
- **SpatialData is read-only.** Images and segmentations are read through `io.read_plate`, and nothing is written back to the SpatialData object.
