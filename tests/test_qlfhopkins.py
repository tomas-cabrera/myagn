import pytest

from myagn.qlfhopkins import qlfhopkins


def test_number_density():
    """
    Check the qlf number density for a few redshifts.
    """
    for z, n in {0.05: 1.250986e-02, 0.5: 7.015659e-03, 5: 1.561466e-04}.items():
        qlf = qlfhopkins.compute(0, z)
        assert qlf["comoving_number_density"][0] == pytest.approx(n, rel=1e-3)


if __name__ == "__main__":
    test_number_density()
