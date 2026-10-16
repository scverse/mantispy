"""Control annotation under spec 10: selection refuses gaps, consumers refuse an ambiguous boolean, annotation keeps classes."""

import numpy as np
import pytest

import mantispy as mt
from mantispy._core.masks import control_annotated, reference_mask


def test_negcon_reference_refuses_missing_control_types(wells):
    """A well the platemap missed must not silently fall out of the reference."""
    types = np.array(wells.obs["Metadata_Control_Type"].astype(object))
    types[np.flatnonzero(types == "negcon")[0]] = None
    wells.obs["Metadata_Control_Type"] = types
    with pytest.raises(ValueError, match="missing value"):
        reference_mask(wells, "negcon")
    with pytest.raises(ValueError, match="missing value"):
        mt.pp.normalize(wells, reference="negcon")


def test_a_bare_control_boolean_is_refused_rather_than_skipped(cells):
    """Under 2.0 the boolean marks any control, so a step that needs the negcon class cannot read it from the flag."""
    del cells.obs["Metadata_Control_Type"]
    with pytest.raises(KeyError, match="annotate_controls"):
        control_annotated(cells)
    with pytest.raises(KeyError, match="annotate_controls"):
        mt.pp.well_qc(cells, max_control_cv=1e-6)


def test_no_control_annotation_at_all_skips_the_control_steps(wells):
    del wells.obs["Metadata_Control_Type"], wells.obs["Metadata_Control"]
    assert control_annotated(wells) is False


def test_annotate_controls_keeps_upstream_poscon_and_empty_classes(wells):
    """A platemap's poscon and empty wells survive marking the negative controls."""
    perturbation = wells.obs["Metadata_Perturbation"].astype(str).to_numpy()
    negcon = perturbation[0]
    upstream = np.where(perturbation == negcon, "negcon", "treatment").astype(object)
    treated = np.flatnonzero(upstream == "treatment")
    upstream[treated[0]], upstream[treated[1]] = "poscon", "empty"
    wells.obs["Metadata_Control_Type"] = upstream
    mt.pp.annotate_controls(wells, negcon=[negcon])
    types = wells.obs["Metadata_Control_Type"].to_numpy()
    assert types[treated[0]] == "poscon"
    assert types[treated[1]] == "empty"
    assert list(wells.obs["Metadata_Control"]) == list(types != "treatment")
    assert mt.io.validate(wells).ok, mt.io.validate(wells).errors


def test_annotate_controls_moves_a_dropped_negcon_to_treatment(wells):
    perturbation = wells.obs["Metadata_Perturbation"].astype(str).to_numpy()
    first, other = perturbation[0], next(p for p in perturbation if p != perturbation[0])
    mt.pp.annotate_controls(wells, negcon=[first])
    mt.pp.annotate_controls(wells, negcon=[other])
    types = wells.obs["Metadata_Control_Type"].to_numpy()
    assert set(types[perturbation == first]) == {"treatment"}
    assert set(types[perturbation == other]) == {"negcon"}


def test_a_staged_builder_keeps_upstream_control_classes():
    """The builders write the spec-10 pair, keeping a plate map's positive and empty controls as controls."""
    import pandas as pd

    from mantispy.ds._build import _control_pair

    types, flag = _control_pair(pd.Series(["negcon", "poscon", "empty", "trt", "compound"]))
    assert list(types) == ["negcon", "poscon", "empty", "treatment", "treatment"]
    assert list(flag) == [True, True, True, False, False]
