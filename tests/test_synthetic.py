import pytest

from mantispy.ds import synthetic_plate


def test_too_many_features_for_the_channel_set_raises():
    with pytest.raises(ValueError, match="at most"):
        synthetic_plate(n_wells=4, n_cells=2, n_features=1000, channels=["only"])
