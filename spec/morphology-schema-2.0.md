# Mantispy canonical morphology schema

Status: experimental draft

Scope: target schema for the current mantispy refactor before the first stable PyPI release

This document defines the data contract that mantispy should build against. It replaces the implicit and partly inconsistent conventions that exist in the current repository. The schema is intentionally not stable yet. Breaking changes are allowed until the implementation, loaders, tutorials, and CellProfiler integrations have been migrated and the maintainers explicitly declare a stable schema version.

The keywords MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY describe requirements for the target implementation.

## 1. Goals

The schema should support one end-to-end model for morphological profiling across three stages:

1. CellProfiler or another measurement system produces object-level measurements and image metadata.
2. Mantispy represents morphology measurements as AnnData and, when image context is retained, links the same rows to SpatialData images and labels.
3. Mantispy aggregates, analyzes, and interprets those measurements while preserving enough identity and provenance to trace results back to source objects and images.

The schema should support classical CellProfiler features, learned embeddings, compound screens, genetic screens, dose-response experiments, single-cell workflows, aggregated profiles, and image-aware interpretation.

The schema should not require every workflow to provide every possible annotation. Common concepts belong in the core schema. Modality-specific concepts are conditional requirements.

## 2. Design principles

### 2.1 Use one source of truth

A semantic value SHOULD have one canonical representation. Mantispy SHOULD NOT keep two synchronized copies of the same information unless an external format requires it.

Examples:

- Cell coordinates live in `Metadata_Center_X` and `Metadata_Center_Y`. Mantispy does not mirror them into `obsm["spatial"]`.
- Source channel names live in `var["channel"]`. Mantispy does not store a second canonical channel column.
- Cell identity comes from explicit metadata columns. Code does not parse `obs_names` to recover identity.
- SpatialData should use an existing mantispy identity column as its instance key when possible rather than creating a duplicate instance identifier.

### 2.2 Store semantics explicitly

Important identity and biological meaning MUST live in named fields. They MUST NOT be encoded only inside `obs_names`, filenames, or concatenated strings.

### 2.3 Keep the global schema small

The global schema defines common morphology concepts. Compound-specific, genetic, dose-response, and other workflow-specific annotations are required only when a workflow uses them.

### 2.4 Preserve external names when they are meaningful

Mantispy SHOULD preserve source identifiers such as channel names and CellProfiler image numbers. Alias handling and matching can happen in helper functions without rewriting the stored source value.

### 2.5 Broad resolution and exact grouping are separate concepts

`resolution` describes the broad statistical level of an AnnData object. `grouped_by` records which metadata columns define one aggregated observation.

### 2.6 Four independent semantic axes

Mantispy keeps four concepts independent, and MUST NOT overload one to encode another. This is a core schema-2.0 design principle.

- **Observation structure** (`resolution`, `grouped_by`) answers: what does one row represent?
- **Feature representation** (`var["feature_kind"]`) answers: what does one column represent?
- **Biological intervention** (`Metadata_Perturbation`, `Metadata_Compound`, `Metadata_Gene`, `Metadata_sgRNA`, `Metadata_Construct`, `Metadata_Allele`, `Metadata_Barcode`) answers: what biological condition or perturbation produced the observation?
- **Experimental identity and context** (`Metadata_Plate`, `Metadata_Well`, `Metadata_Batch`, `Metadata_Source`, `Metadata_ImageID`, `Metadata_ObjectNumber`) answers: where did the observation come from?

A gene aggregate, for example, is `resolution="aggregate"` with `grouped_by=["Metadata_Gene"]` (structure), while the gene it targets lives in `Metadata_Gene` (intervention) and the plate it came from in `Metadata_Plate` (context). The resolution never becomes `gene`.

## 3. AnnData core contract

A mantispy AnnData object MUST follow the rules below.

### 3.1 `X`

`X` contains measurement features only.

Requirements:

- `X` MUST be numeric.
- `X` MUST use `float32` in mantispy-written objects.
- Metadata, identifiers, counts, coordinates, labels, and result columns MUST NOT be stored in `X`.
- CellProfiler measurements, learned embeddings, and derived feature matrices may all live in `X`, but `var["feature_kind"]` MUST distinguish their type.

Sparse input MAY be accepted by readers and functions, but the schema does not require sparse storage.

### 3.2 `obs`

`obs` stores observation identity, experimental annotations, biological annotations, and observation-level analysis outputs.

#### Metadata namespace

`Metadata_<Concept>` is used for observation-level identity, experimental context, provenance, or biological annotation, whether read from upstream or computed by mantispy. Examples include `Metadata_Plate`, `Metadata_Well`, `Metadata_Batch`, `Metadata_Source`, `Metadata_Compound`, `Metadata_Concentration`, `Metadata_Gene`, `Metadata_sgRNA`, `Metadata_Construct`, `Metadata_Allele`, `Metadata_Barcode`, `Metadata_ImageID`, and `Metadata_ObjectNumber`.

Quantitative analysis results MUST NOT use this namespace. They use lowercase snake_case names such as `qc_pass`, `hits_qvalue`, `effect`, `cluster`, and `map`. The boundary is annotation versus result: a column describing where an observation came from or what condition produced it is `Metadata_`; a number mantispy computed from the data is not.

The prefix is a naming convention. It MUST NOT be used as the sole rule for deciding which columns survive aggregation, are passed to metrics, or are exported.

String metadata MAY use pandas categorical dtype for storage efficiency, but analysis semantics MUST NOT depend on whether a column is `str`, `object`, or `category`, unless the function explicitly requires a categorical variable.

### 3.3 `var`

Every mantispy object MUST contain the existing feature annotation columns:

- `object`
- `feature_group`
- `feature`
- `channel`
- `scale`
- `angle`
- `gray_levels`
- `radial_bin`
- `params`
- `is_feature`

The schema adds one required column:

- `feature_kind`

`feature_kind` MUST use one of the following values:

- `measurement`: a directly measured morphology feature, including CellProfiler-style measurements
- `embedding`: a learned representation dimension
- `derived`: a feature produced by a mantispy transformation such as composition, signature, or trajectory construction

`feature_kind` is required and non-null for every feature in a valid schema-2.0 object: a null is as much an error as a value outside the vocabulary. An in-package producer sets it when it knows the representation; the name parser tags CellProfiler measurements, and the embedding and derived producers tag their output. Where mantispy cannot know the representation of an externally built object, the caller declares it through `mt.io.stamp(..., feature_kind=...)`; mantispy never infers it from feature names.

The ten existing annotation fields MAY be empty when they do not apply. For example, an embedding dimension can have `feature_kind="embedding"` while CellProfiler-specific fields are empty.

`var_names` remain the feature identifiers and MUST be unique.

### 3.4 `layers`, `obsm`, `varm`, and `obsp`

These slots remain available for tool outputs and alternative representations.

The schema does not require a universal naming system for all representations in this draft, but public functions MUST document which slot they read and write.

Mantispy MUST NOT store cell coordinates in `obsm["spatial"]` as a second copy of `Metadata_Center_X` and `Metadata_Center_Y`.

### 3.5 `uns["mantispy"]`

All mantispy-owned object metadata MUST live under `uns["mantispy"]`.

The namespace MUST contain:

- `schema_version`
- `schema_status`
- `resolution`
- `grouped_by`
- `history`

During the current refactor, `schema_status` MUST be `"experimental"`.

A stable schema version MUST NOT be declared until the package implementation, loaders, staged datasets, tutorials, and CellProfiler integrations follow the contract.

## 4. Observation resolution

`uns["mantispy"]["resolution"]` MUST be explicitly present. Mantispy MUST NOT infer a missing resolution differently in different code paths.

Allowed values are:

- `object`
- `well`
- `aggregate`

These are a structural hierarchy, not a biological ontology. They answer "what does one row represent?" at the level of structure; the exact unit is carried by `grouped_by`.

### 4.1 Object resolution

One row represents one primary segmented object. That object may be a CellProfiler Cells, Nuclei, Cytoplasm, or another primary segmented object; `object` is not a biological-cell claim and is not a feature representation.

For object-level data:

```python
resolution = "object"
grouped_by = []
```

### 4.2 Well resolution

One row represents one physical well-level profile.

For standard plate-based profiling:

```python
resolution = "well"
grouped_by = ["Metadata_Plate", "Metadata_Well"]
```

### 4.3 Aggregate resolution

One row represents an aggregate above object/well level. The exact row semantics MUST be carried by `uns["mantispy"]["grouped_by"]`.

Examples:

```python
resolution = "aggregate"
grouped_by = ["Metadata_Gene"]
```

```python
resolution = "aggregate"
grouped_by = ["Metadata_sgRNA"]
```

```python
resolution = "aggregate"
grouped_by = ["Metadata_Compound", "Metadata_Concentration"]
```

```python
resolution = "aggregate"
grouped_by = ["Metadata_MOA"]
```

Guide, gene, construct, compound, dose, signature, and other aggregate types do not get new resolution values. Mantispy MUST NOT add biological resolution values such as `gene`, `guide`, `compound`, or `construct`; `grouped_by` carries that information.

### 4.4 The structural model across workflows

The three tiers, with `grouped_by` and `feature_kind`, represent every intended modality without new resolution values:

- Drug: `well` → `aggregate` grouped by `Metadata_Compound` + `Metadata_Concentration`.
- Arrayed CRISPR: `object` → `well` → `aggregate` grouped by `Metadata_sgRNA` or `Metadata_Gene`.
- Pooled CRISPR / optical pooled screening: `object` → `aggregate` grouped by `Metadata_Barcode`, `Metadata_sgRNA`, or `Metadata_Gene`.
- ORF: `object` → `well` → `aggregate` grouped by `Metadata_Construct` or `Metadata_Gene`.
- Unperturbed single-object morphology: `object` → `well`, or a derived `aggregate`.
- Learned representations: any of `object`, `well`, or `aggregate`, with `feature_kind="embedding"`.

Biological modality and feature representation are orthogonal to the resolution tier.

## 5. Observation identity

Semantic identity MUST live in explicit metadata fields.

`obs_names` MUST be unique strings, but their contents are not part of the schema. Mantispy functions MUST NOT parse `obs_names` to recover plate, well, image, object, perturbation, or other meaning.

Importers MAY generate deterministic `obs_names` for convenience. Those names remain row identifiers rather than semantic fields.

### 5.1 Plate identity

`Metadata_Plate` is the canonical plate identifier.

Requirements:

- `Metadata_Plate` MUST be globally unique within data that may be combined or analyzed together.
- Importers and exporters are responsible for namespacing source plate barcodes when collisions are possible.
- Mantispy does not define a second `Metadata_PlateID` field.

`Metadata_Well` identifies a well within a plate and MUST use the normalized well format already used by mantispy, such as `A01`.

A well is identified by:

```text
Metadata_Plate + Metadata_Well
```

### 5.2 Image identity

`Metadata_ImageID` is the canonical globally unique identifier for one field or image set.

Requirements:

- `Metadata_ImageID` MUST be globally unique within data that may be combined.
- Consumers MUST treat the value as opaque and MUST NOT parse semantic information from it.
- Importers SHOULD generate the identifier deterministically when the source provides stable field identity.
- `Metadata_ImageNumber` SHOULD preserve the original CellProfiler image number when one exists.
- `Metadata_Site` SHOULD preserve the source site or field number when one exists.

The schema does not require a global experiment identifier. Global uniqueness of plate and image identifiers is the responsibility of the importer or exporter.

### 5.3 Object identity

One object-level row represents one primary object.

Required object-level fields are:

- `Metadata_ImageID`
- `Metadata_ObjectType`
- `Metadata_ObjectNumber`

`Metadata_ObjectType` names the primary object set, for example `Cells` or `Cytoplasm`.

`Metadata_ObjectNumber` is the instance number of that primary object within `Metadata_ImageID`.

A primary object is identified by:

```text
Metadata_ImageID + Metadata_ObjectNumber
```

The pair MUST be unique in a object-level AnnData object.

### 5.4 Related object identities

Mantispy keeps one row per primary object. Features from related objects such as nuclei and cytoplasm may be joined onto that row.

When related object instance numbers are available, importers MUST preserve them rather than dropping them.

Related object IDs use the pattern:

```text
Metadata_<ObjectName>ObjectNumber
```

Examples include:

- `Metadata_NucleiObjectNumber`
- `Metadata_CytoplasmObjectNumber`
- `Metadata_CellsObjectNumber`

The column corresponding to the primary object MAY be omitted to avoid duplicating `Metadata_ObjectNumber`.

Mantispy MUST NOT assume that Cells, Nuclei, and Cytoplasm use the same instance number.

## 6. Coordinates

Cell centroid coordinates live in:

- `Metadata_Center_X`
- `Metadata_Center_Y`

For CellProfiler data, these values represent pixel coordinates in the local field or image frame.

Mantispy MUST NOT maintain a duplicate canonical copy in `obsm["spatial"]`.

Functions that require coordinates MUST validate the presence of these fields.

SpatialData owns transformations between image and global coordinate systems when a SpatialData object is used.

## 7. SpatialData linkage

A mantispy AnnData table stored inside SpatialData MUST use SpatialData table annotation semantics.

Mantispy uses:

- `region` as the SpatialData region key column
- `Metadata_ObjectNumber` as the SpatialData instance key for the primary object table

The table MUST be parsed or validated with the equivalent of:

```python
TableModel.parse(
    adata,
    region=<annotated label element or elements>,
    region_key="region",
    instance_key="Metadata_ObjectNumber",
)
```

`region` identifies the label or shape element annotated by each row. `Metadata_ObjectNumber` contains the corresponding instance value in that element.

Mantispy does not require a duplicate `instance_id` column because `Metadata_ObjectNumber` already contains the required identity.

For every linked row:

```text
region + Metadata_ObjectNumber
```

MUST resolve to the intended primary object instance.

If related compartments have separate label elements, their object numbers are preserved through the related object ID columns described above. Support for directly annotating multiple compartment label elements can be added later without changing the primary cell identity model.

## 8. Image metadata

When image-level metadata are retained in AnnData without a full SpatialData object, mantispy stores them in:

```python
uns["mantispy"]["image_table"]
```

If present, the image table MUST use `Metadata_ImageID` as its unique index or unique key.

It MAY contain:

- `Metadata_Plate`
- `Metadata_Well`
- `Metadata_Site`
- `Metadata_ImageNumber`
- CellProfiler image quality measurements
- source file paths or URLs
- channel-specific source references

Every object-level `Metadata_ImageID` referenced by `obs` MUST resolve to exactly one image-table row when an image table is present.

Image QC and other image-level operations MUST join through `Metadata_ImageID`, not through `Metadata_ImageNumber` alone.

## 9. Perturbation semantics

`Metadata_Perturbation` is the canonical replicate-group identifier.

Rows with the same `Metadata_Perturbation` value are intended to represent biological replicates of the same perturbation unit for analyses that use this field as their grouping key.

The exact biological meaning may vary by workflow. For example, a perturbation can represent a compound-dose condition, gene, guide, construct, or another replicate unit.

The schema does not require all biological meaning to be encoded inside `Metadata_Perturbation`.

Workflow-specific fields remain explicit when available, for example:

- `Metadata_Compound`
- `Metadata_Concentration`
- `Metadata_Gene`
- `Metadata_sgRNA`
- `Metadata_Construct`
- `Metadata_Allele`

Functions that require one of these concepts MUST request or validate the corresponding field instead of parsing `Metadata_Perturbation`.

## 10. Control semantics

The control model changes from the current mantispy convention.

### 10.1 `Metadata_Control`

`Metadata_Control` is a required boolean wherever control status is defined.

It means:

```text
True  = the observation is a control
False = the observation is a treatment or non-control
```

`Metadata_Control` no longer means negative control only.

### 10.2 `Metadata_Control_Type`

When `Metadata_Control` is present, `Metadata_Control_Type` MUST also be present.

Allowed values are:

- `negcon`
- `poscon`
- `treatment`
- `empty`

The expected relationship is:

| Metadata_Control_Type | Metadata_Control |
| --- | --- |
| `negcon` | `True` |
| `poscon` | `True` |
| `empty` | `True` |
| `treatment` | `False` |

Dataset-specific labels such as `non-targeting`, `nontargeting`, `no-guide`, `intergenic`, `DMSO`, or specific positive-control compounds do not belong in the control-type vocabulary. They remain in the relevant biological annotation fields.

### 10.3 Reference selection

The mantispy reference resolver MUST use `Metadata_Control_Type` for named control classes.

In particular:

```python
reference = "negcon"
```

selects rows where:

```python
adata.obs["Metadata_Control_Type"] == "negcon"
```

It MUST NOT interpret the `Metadata_Control` boolean alone as negative-control membership.

The same rule applies to `poscon` if that reference mode is supported.

## 11. Conditional workflow schemas

The fields below are conditional requirements. They are not part of the minimum schema for every mantispy object.

### 11.1 Compound workflows

Compound-specific tools require the fields they consume.

Common fields are:

- `Metadata_Compound`
- `Metadata_MOA`
- `Metadata_Concentration`
- `Metadata_ConcentrationUnit`

`Metadata_MOA` is optional unless an MOA-specific function requires it.

### 11.2 Dose-response workflows

If `Metadata_Concentration` exists, `Metadata_ConcentrationUnit` MUST also exist.

`Metadata_Concentration` is numeric.

`Metadata_ConcentrationUnit` uses an explicit unit. Recommended values are:

- `M`
- `mM`
- `uM`
- `nM`
- `pM`

Mantispy MUST NOT silently assume a concentration unit.

Dose-response functions MUST either convert supported units explicitly or require a consistent unit and raise a clear error when units differ.

A CRISPR, ORF, or other genetic experiment does not need concentration fields unless the experiment genuinely has a dose variable.

### 11.3 Genetic workflows

Genetic tools use explicit fields rather than overloading `Metadata_Perturbation`.

Common fields are:

- `Metadata_Gene`
- `Metadata_sgRNA`
- `Metadata_Construct`
- `Metadata_Allele`
- `Metadata_ChromosomeArm`

A guide-level object can therefore use:

```text
Metadata_Perturbation = guide identifier
Metadata_sgRNA = guide identifier
Metadata_Gene = target gene
```

while a gene-level aggregate uses:

```python
resolution = "aggregate"
grouped_by = ["Metadata_Gene"]
```

### 11.4 Learned embeddings

Learned embedding dimensions use:

```text
feature_kind = "embedding"
```

The CellProfiler-specific `var` annotations MAY be empty.

Tools that require morphology feature families, channels, or CellProfiler-specific feature semantics MUST reject or clearly skip embedding features rather than inferring nonexistent annotations.

## 12. Channel semantics

`var["channel"]` stores the source channel name.

Mantispy MUST NOT rewrite the stored value into a second canonical vocabulary during ingestion.

Examples such as `BFHigh` and `Brightfield_H` may therefore remain different stored values if that is what the source data use.

Functions that need to compare profile channels with image channels SHOULD accept an explicit alias mapping or use a lookup helper. The mapping MUST NOT create a second stored source of truth.

## 13. Aggregation semantics

### 13.1 Metadata preservation

When observations are aggregated, mantispy MUST preserve every `obs` column that is constant within each aggregation group, regardless of whether the column starts with `Metadata_`.

A column that varies within a group MUST NOT be silently copied to the aggregate unless the operation defines an explicit aggregation rule for that column.

The `Metadata_` prefix MUST NOT act as an aggregation-survival filter.

### 13.2 Resolution and grouping

`tl.aggregate` and other level-changing operations MUST write the resulting `resolution` and `grouped_by` explicitly.

For standard cell-to-well aggregation:

```python
resolution = "well"
grouped_by = ["Metadata_Plate", "Metadata_Well"]
```

For higher-level aggregation, `resolution` is `"aggregate"` and `grouped_by` records the actual grouping columns.

### 13.3 Counts

The existing count names remain canonical:

- `Metadata_CellCount`
- `Metadata_SiteCount`
- `Metadata_ReplicateCount`

Functions SHOULD write the relevant count when the source data allow it.

### 13.4 Optional source membership

Aggregation MAY retain explicit source membership when the caller requests it.

Membership storage is optional because object-level membership can be large.

When membership is stored, it MUST reference stable source identity fields rather than integer row positions.

For cell-to-profile aggregation, source members are identified by:

```text
Metadata_ImageID + Metadata_ObjectNumber
```

The target implementation SHOULD store membership under:

```python
uns["mantispy"]["membership"]
```

with enough metadata to state:

- the source resolution
- the source identity columns
- the target observation row
- the source identities belonging to that target

The exact physical representation may be optimized during implementation, but it MUST NOT rely on source row positions.

## 14. Transformation provenance

Mantispy keeps lightweight portable provenance inside the AnnData object.

Full environment, file, notebook, and artifact lineage belongs in external workflow systems such as LaminDB and is not a hard mantispy dependency.

### 14.1 History

`uns["mantispy"]["history"]` is an ordered append-only list.

Each mantispy operation that changes data or creates a persistent result SHOULD append one record.

A history record contains at least:

```python
{
    "operation": "pp.normalize",
    "params": {...},
    "input": "X",
    "output": "X",
    "mantispy_version": "...",
}
```

For operations that change resolution, the record SHOULD also include the source and target resolution or grouping information.

The order of records in the list defines execution order. Timestamps are optional.

### 14.2 No overwrite-by-function behavior

Running the same operation twice MUST append two history records. A later call MUST NOT erase the earlier call's provenance.

Result-specific parameters SHOULD be stored with the corresponding result when a plot or downstream function needs them.

### 14.3 Level-changing operations

A new AnnData object created by aggregation, consensus, signature construction, or another level-changing operation MUST inherit the source object's history and append its own operation.

The new object MUST also preserve relevant mantispy descriptors unless the constructor intentionally changes them.

## 15. Result table schema

All persistent mantispy result tables MUST follow one naming convention.

Result tables remain stored under `uns["mantispy"]` in this draft. Moving them into another namespace is not required by this schema.

### 15.1 Common column names

When a concept exists, the following names are canonical:

- `group`: primary tested or summarized group
- `reference`: comparison or reference group, when applicable
- `feature`: feature identifier, when applicable
- `n_obs`: number of observations contributing to the result
- `effect`: effect estimate when the quantity is an effect
- `distance`: distance when the quantity is genuinely a distance
- `score`: method-specific score when neither effect nor distance is the correct semantic name
- `pvalue`: raw p-value
- `qvalue`: multiple-testing adjusted p-value
- `is_significant`: generic significance decision, when applicable
- `is_hit`: hit-calling decision, when the result is specifically a hit call

Mantispy MUST NOT use alternate spellings such as `p_value` or `corrected_p_value` for these concepts.

Tool-specific columns are allowed when they represent genuinely different quantities, such as `ec50`, `hill_slope`, or `r_squared`.

### 15.2 Tidy structure

A result table SHOULD contain one logical result per row.

Group identity MUST be represented in columns. Result consumers MUST NOT recover semantic group identity by parsing a DataFrame index.

### 15.3 Registration

Every persistent result key written under `uns["mantispy"]` MUST be registered in the schema registry used by validation and tests.

The implementation MUST NOT maintain a declared result-key list that is ignored by runtime behavior.

## 16. Validation

Validation happens at two levels.

### 16.1 Full validation at IO boundaries

Mantispy MUST run full schema validation when:

- reading or constructing a mantispy object through public mantispy IO
- explicitly stamping an external AnnData as a mantispy object
- writing a mantispy object

Full validation checks the global contract and the requirements implied by `resolution`.

### 16.2 Targeted validation in public functions

Public analysis functions MUST validate only the fields and invariants they need.

Examples:

- a dose-response function checks concentration and concentration unit
- a cell-density function checks object resolution and centroid coordinates
- a genetic function checks the required gene or guide field
- an image-aware function checks `Metadata_ImageID` and object linkage

Public functions SHOULD NOT run an expensive full-object validation on every call.

### 16.3 Errors and warnings

Missing core identity or incompatible semantics SHOULD raise clear errors.

Optional metadata SHOULD produce warnings only when the function can still produce a scientifically valid result without it.

## 17. Minimum schemas by resolution

### 17.1 Object-level minimum

An object-level mantispy object requires:

- `X` and the required `var` columns
- `Metadata_Plate`
- `Metadata_Well`
- `Metadata_ImageID`
- `Metadata_ObjectType`
- `Metadata_ObjectNumber`
- explicit `resolution="object"`
- `grouped_by=[]`

The following are conditional:

- `Metadata_Site`
- `Metadata_ImageNumber`
- `Metadata_Center_X`
- `Metadata_Center_Y`
- related object-number columns
- control fields
- perturbation fields

### 17.2 Well-level minimum

A standard well-level mantispy object requires:

- `X` and the required `var` columns
- `Metadata_Plate`
- `Metadata_Well`
- explicit `resolution="well"`
- `grouped_by=["Metadata_Plate", "Metadata_Well"]`

`Metadata_CellCount` and `Metadata_SiteCount` SHOULD be present when known.

### 17.3 Aggregate-level minimum

An aggregate-level object requires:

- `X` and the required `var` columns
- explicit `resolution="aggregate"`
- non-empty `grouped_by`
- the columns listed in `grouped_by`

The exact biological fields depend on the workflow.

## 18. CellProfiler import requirements

A CellProfiler-to-AnnData importer or plugin targeting this schema MUST preserve enough information to construct the core cell identity without relying on implicit number equality between object types.

It MUST preserve:

- canonical plate and well
- globally unique `Metadata_ImageID`
- original `Metadata_ImageNumber` when available
- site or field identifier when available
- primary object type
- primary object number
- related object numbers when available
- center coordinates when available
- morphology measurements
- source channel names
- control and perturbation metadata when supplied by the pipeline or annotation input

It SHOULD preserve:

- image source paths or URLs
- image quality measurements
- CellProfiler pipeline provenance

The importer MUST NOT place parent, child, neighbour object identifiers, file paths, counts, or other non-measurement numeric metadata into `X`.

## 19. SpatialData import and export requirements

A CellProfiler-to-SpatialData path MUST produce a mantispy-compatible AnnData table and valid SpatialData table linkage.

At minimum, the SpatialData object SHOULD contain:

- source images
- primary-object labels
- a mantispy AnnData table containing morphology measurements

The AnnData table MUST satisfy this schema before it is added as a SpatialData table.

The table MUST link to the primary-object labels using `region` and `Metadata_ObjectNumber` as described above.

Images, labels, and tables MUST share valid SpatialData coordinate transformations. Mantispy does not duplicate those transformations inside AnnData metadata.

## 20. Migration requirements for the current package

The current implementation must be migrated before this schema can be declared stable.

The migration includes at least the following changes:

1. Remove ambiguous unstamped resolution behavior and write `grouped_by` consistently.
2. Stop treating `obs_names` as semantic identity inside analysis functions.
3. Add globally unique `Metadata_ImageID` and use it for image joins.
4. Add `Metadata_ObjectType` and preserve related object instance numbers.
5. Unify SpatialData linkage on one region and instance-key convention.
6. Migrate control semantics so `Metadata_Control` means any control and `Metadata_Control_Type` carries the fixed type vocabulary.
7. Preserve all constant observation columns during aggregation rather than using the `Metadata_` prefix as a survival filter.
8. Add optional source membership for aggregation without relying on row positions.
9. Replace overwrite-by-function provenance with append-only `history` and preserve history through level changes.
10. Add `feature_kind` to the required feature schema.
11. Preserve source channel names without ingest-time canonicalization.
12. Require an explicit concentration unit whenever concentration is stored.
13. Standardize persistent result-table column names and register every persistent result key.
14. Run full validation at mantispy IO boundaries and targeted validation inside public functions.
15. Rebuild staged datasets and update tutorials after the code migration.

## 21. Stability policy

The schema is experimental until the maintainers explicitly declare it stable.

Before stability:

- breaking schema changes are allowed
- staged datasets may be rebuilt
- loaders and APIs may be migrated without compatibility shims when that produces a cleaner first stable contract

After stability:

- schema versions follow semantic versioning
- backward-compatible additions use a minor version
- removals, renames, or semantic changes require a major version
- patch versions fix validation or documentation without changing the data contract

The README should summarize the stable schema only after the implementation follows it. The detailed contract belongs in `spec/` and should remain the authoritative source.

## 22. Acceptance criteria for the schema refactor

The schema refactor is complete when all of the following are true:

1. A CellProfiler cell can be uniquely identified without parsing `obs_names`.
2. The same cell can be linked to the correct SpatialData primary-object label instance.
3. Related nucleus, cytoplasm, and cell object IDs are preserved when they exist.
4. A cell can be traced to a globally unique image identifier.
5. Cell-to-well aggregation records the correct resolution and grouping and preserves constant metadata.
6. Optional membership can identify the source cells of a profile using stable identity fields.
7. A perturbation aggregate states exactly which columns define one row through `grouped_by`.
8. Negative and positive controls are unambiguous under the new control model.
9. Dose-response data cannot be analyzed without explicit concentration units.
10. Repeated mantispy operations preserve an ordered history rather than overwriting provenance.
11. CellProfiler measurements, embeddings, and derived features can be distinguished through `feature_kind`.
12. Persistent result tables use the standardized naming rules.
13. Full validation passes for the canonical compound, genetic, single-cell, embedding, and SpatialData fixtures.
14. The public tutorials and staged datasets follow the same schema as the implementation.

Only after these criteria are met should the maintainers choose and publish the first stable schema version.
