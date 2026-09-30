"""Plots for judging profile strength and correction quality."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core.frames import as_frame
from mantispy.metrics._common import embedding, r_squared
from mantispy.pl._common import axes as _axes
from mantispy.pl._common import maybe_interactive as _maybe_interactive
from mantispy.pl._common import returned as _returned
from mantispy.pl._common import table as _table

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def map(adata: AnnData, key: str = "map", label_top: int = 10, ax: Axes | None = None) -> Axes | None:
    """Mean average precision against significance, with the strongest groups labeled.

    The dashed line is the significance threshold the run used, so a point above it and to the right is a perturbation that is both strong and reproducible.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.map` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        label_top: How many of the strongest groups to label, taken by mean average precision.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold one point per group and the threshold drawn as a labeled reference line.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
    """
    table = _table(adata, key, "mt.tl.map")
    ax = _axes(ax, (5.5, 4.5))

    significance = -np.log10(np.clip(table["corrected_p_value"].to_numpy(dtype=float), 1e-12, None))
    ax.scatter(table["mean_average_precision"], significance, s=14)

    # Params are stored under the function name, not the table's `key`.
    threshold = adata.uns.get("mantispy", {}).get("params", {}).get("map", {}).get("threshold", 0.05)
    ax.axhline(-np.log10(threshold), color="grey", ls="--", lw=1, label=f"q = {threshold}")

    group_column = table.columns[0]
    for _, row in table.nlargest(label_top, "mean_average_precision").iterrows():
        ax.annotate(
            str(row[group_column]),
            (row["mean_average_precision"], -np.log10(max(float(row["corrected_p_value"]), 1e-12))),
            fontsize=6,
        )
    ax.set_xlabel("mean average precision")
    ax.set_ylabel("-log10 corrected p")
    ax.legend(fontsize=7)

    tidy = pd.DataFrame(
        {
            group_column: table[group_column].astype(str).to_numpy(),
            "mean average precision": table["mean_average_precision"].to_numpy(dtype=float),
            "-log10 corrected p": significance,
        }
    )
    _maybe_interactive(
        "scatter",
        ax=ax,
        data=tidy,
        x="mean average precision",
        y="-log10 corrected p",
        hover=[group_column],
        title="mean average precision",
    )
    return _returned(ax)


def replicate_correlation(adata: AnnData, key: str = "percent_replicating", ax: Axes | None = None) -> Axes | None:
    """Observed replicate correlation against each group's permutation threshold.

    Points above the diagonal replicate; the distance from it is the margin.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.percent_replicating` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold the groups that replicate colored apart from those that do not and the diagonal drawn as a reference line.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
    """
    table = _table(adata, key, "mt.tl.percent_replicating")
    ax = _axes(ax, (5, 4.5))

    replicating = table["is_replicating"].to_numpy(dtype=bool)
    ax.scatter(
        table["null_threshold"][~replicating], table["median_replicate_correlation"][~replicating], s=14, label="no"
    )
    ax.scatter(
        table["null_threshold"][replicating],
        table["median_replicate_correlation"][replicating],
        s=14,
        color="seagreen",
        label="yes",
    )
    limits = [
        float(np.nanmin([table["null_threshold"].min(), table["median_replicate_correlation"].min()])),
        float(np.nanmax([table["null_threshold"].max(), table["median_replicate_correlation"].max()])),
    ]
    ax.plot(limits, limits, color="grey", ls="--", lw=1)
    ax.set_xlabel("null threshold")
    ax.set_ylabel("median replicate correlation")
    ax.legend(title="replicating", fontsize=7, title_fontsize=7)

    group_column = table.columns[0]
    tidy = pd.DataFrame(
        {
            group_column: table[group_column].astype(str).to_numpy(),
            "null threshold": table["null_threshold"].to_numpy(dtype=float),
            "median replicate correlation": table["median_replicate_correlation"].to_numpy(dtype=float),
            "replicating": np.where(replicating, "yes", "no"),
        }
    )
    _maybe_interactive(
        "scatter",
        ax=ax,
        data=tidy,
        x="null threshold",
        y="median replicate correlation",
        color="replicating",
        hover=[group_column],
        title="replicate correlation",
    )
    return _returned(ax)


def batch_variance(
    adata: AnnData,
    keys: Sequence[str],
    use_rep: str = "X_pca",
    n_comps: int | None = None,
    ax: Axes | None = None,
) -> Axes | None:
    """R^2 of each principal component on each covariate.

    A covariate with high R^2 in the leading components accounts for much of the embedding's structure.

    Args:
        adata: Object with the embedding to measure in.
        keys: ``obs`` columns to score, one line each.
        use_rep: ``obsm`` key of the embedding.
        n_comps: Draw only the leading components, or ``None`` for every component the embedding holds.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold one line per entry of ``keys`` and the y axis fixed to ``[0, 1]``.

    Raises:
        KeyError: ``obsm`` holds nothing under ``use_rep``, or ``obs`` has no column for one of ``keys``.
    """
    values = embedding(adata, use_rep)
    if n_comps is not None:
        values = values[:, :n_comps]

    ax = _axes(ax, (6, 4))
    components = np.arange(1, values.shape[1] + 1)
    records = []
    for key in keys:
        covariate = as_frame(adata.obs)[key]
        explained = [r_squared(values[:, index], covariate) for index in range(values.shape[1])]
        ax.plot(components, explained, marker="o", ms=3, label=key)
        records.append(
            pd.DataFrame({"principal component": components, "variance explained (R²)": explained, "covariate": key})
        )
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (R²)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

    if records:
        _maybe_interactive(
            "line",
            ax=ax,
            data=pd.concat(records, ignore_index=True),
            x="principal component",
            y="variance explained (R²)",
            color="covariate",
            title="variance explained per component",
        )
    return _returned(ax)


def metrics(table: pd.DataFrame, ax: Axes | None = None) -> Axes | None:
    """Grouped bars of a tidy metrics table, one group of bars per metric and one bar per representation.

    Takes the table instead of an AnnData because the table already holds every representation side by side.

    Args:
        table: A tidy frame with ``metric``, ``representation`` and ``value``, as :func:`~mantispy.metrics.known_relationships` returns and as :func:`~mantispy.metrics.batch_variance_explained` rows can be stacked into.
            Its ``better`` column, when present, adds the direction that is an improvement to each tick label.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold one group of bars per metric and one bar per representation.

    Raises:
        ValueError: The table holds more than one value for some metric and representation, which cannot be pivoted into a grid.
    """
    pivot = table.pivot(index="metric", columns="representation", values="value")
    ax = _axes(ax, (7, 4))
    pivot.plot.bar(ax=ax)

    if "better" in table.columns:
        # Covariate rows carry no direction, so their tick keeps the bare metric name.
        direction = table.drop_duplicates("metric").set_index("metric")["better"]
        labels = [
            f"{name}\n({direction[name]} is better)" if isinstance(direction.get(name), str) else name
            for name in pivot.index
        ]
        ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("value")
    ax.set_xlabel("")
    ax.legend(fontsize=7)

    tidy = pivot.reset_index().melt(id_vars="metric", var_name="representation", value_name="value")
    _maybe_interactive(
        "barh",
        ax=ax,
        data=tidy,
        x="value",
        y="metric",
        color="representation",
        barmode="group",
        title="correction metrics",
    )
    return _returned(ax)


def _integration_heatmap(
    frame: pd.DataFrame,
    blocks: dict[str, list[str]],
    *,
    aggregate_block: str = "Aggregate",
    ax: Axes | None = None,
) -> Axes | None:
    """The integration-benchmark heatmap :func:`~mantispy.metrics.evaluate_integration` draws.

    Draws one row per representation and one column per metric, the columns grouped into blocks that are
    separated by a gap and carry a centered header. The metric blocks use the purple-green ``PRGn`` map and
    the aggregate block a distinct ``YlGnBu``, so summary columns read apart from the metrics they summarize.
    Every cell is annotated with its value, with the text colour chosen for contrast against the cell.

    Args:
        frame: Numeric results, one row per representation and one column per metric or aggregate score.
        blocks: Ordered mapping of block header to the columns it holds, left to right; a column missing from
            ``frame`` is drawn grey and left blank.
        aggregate_block: Which block header gets the distinct aggregate colormap.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    reps = [str(name) for name in frame.index]
    n_rows = len(reps)

    gap = 0.6
    left_of: dict[str, float] = {}
    spans: list[tuple[str, float, float]] = []
    cursor = 0.0
    for index, (title, columns) in enumerate(blocks.items()):
        if index:
            cursor += gap
        start = cursor
        for column in columns:
            left_of[column] = cursor
            cursor += 1.0
        spans.append((title, start, cursor))
    width = cursor

    ax = _axes(ax, (max(width * 0.72 + 1.5, 4.0), n_rows * 0.55 + 2.2))
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    metric_cmap = plt.get_cmap("PRGn")
    aggregate_cmap = plt.get_cmap("YlGnBu")

    for title, _start, _end in spans:
        cmap = aggregate_cmap if title == aggregate_block else metric_cmap
        for column in blocks[title]:
            left = left_of[column]
            for row, rep in enumerate(reps):
                value = float(frame.loc[rep, column]) if column in frame.columns else float("nan")
                if np.isnan(value):
                    ax.add_patch(Rectangle((left, row), 1.0, 1.0, facecolor="0.9", edgecolor="white", linewidth=1.0))
                    continue
                colour = cmap(norm(value))
                ax.add_patch(Rectangle((left, row), 1.0, 1.0, facecolor=colour, edgecolor="white", linewidth=1.0))
                luminance = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
                ax.text(
                    left + 0.5,
                    row + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if luminance < 0.5 else "black",
                )

    for title, start, end in spans:
        ax.text((start + end) / 2.0, -0.5, title, ha="center", va="bottom", fontsize=8, fontweight="bold")
    for column, left in left_of.items():
        ax.text(left + 0.5, n_rows + 0.15, column, ha="right", va="top", rotation=45, fontsize=7)

    ax.set_yticks([row + 0.5 for row in range(n_rows)])
    ax.set_yticklabels(reps, fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(n_rows + 0.2, -1.3)  # inverted, so the first representation is on top with room for the headers
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _returned(ax)


def similarity(
    adata: AnnData,
    key: str = "similarity",
    groupby: str | None = "Metadata_Perturbation",
    max_obs: int = 500,
    ax: Axes | None = None,
) -> Axes | None:
    """Profile-by-profile similarity, ordered by ``groupby`` so blocks are visible.

    Subsamples to ``max_obs`` rows with a fixed seed when the object is larger, because the matrix is quadratic.

    Args:
        adata: Object holding the matrix :func:`~mantispy.tl.similarity` wrote in ``obsp``.
        key: Name of that matrix in ``obsp``.
        groupby: ``obs`` column to order the rows and columns by, or ``None`` to keep the object's own order.
        max_obs: Draw at most this many profiles, sampled from the ordering rather than from the rows.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold the ordered matrix on a diverging scale fixed to ``[-1, 1]`` with a colorbar beside it.

    Raises:
        KeyError: ``obsp`` holds nothing under ``key``.
    """
    if key not in adata.obsp:
        raise KeyError(f"obsp has no {key!r}; run mt.tl.similarity first")
    matrix = np.asarray(adata.obsp[key])

    order = np.arange(adata.n_obs)
    if groupby is not None:
        order = np.argsort(adata.obs[groupby].astype(str).to_numpy(), kind="stable")
    if order.size > max_obs:
        # Sample positions within the ordering, not row indices, so the groupby blocks stay contiguous.
        picked = np.sort(np.random.default_rng(0).choice(order.size, size=max_obs, replace=False))
        order = order[picked]

    ax = _axes(ax, (6, 5))
    ordered = matrix[np.ix_(order, order)]
    image = ax.imshow(ordered, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{key} ({order.size} profiles)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(image, ax=ax, fraction=0.045, label="similarity")

    profiles = adata.obs_names[order].astype(str).tolist()
    _maybe_interactive(
        "heatmap",
        ax=ax,
        matrix=ordered,
        rows=profiles,
        columns=profiles,
        value_label="similarity",
        title=f"{key} ({order.size} profiles)",
    )
    return _returned(ax)
