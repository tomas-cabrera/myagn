"""Reproduce AGN components of Figure 1 from Palmese+21.
The original figure includes information from the skymap for GW190521,
so the resulting figure from this script will not be identical."""

import os
import os.path as pa

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from myagn.distributions import QLFHopkins
from myagn.qlfhopkins import qlfhopkins

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def test_palmese21():
    """Reproduce Figure 1 from Palmese+21."""

    # Initialize distribution
    qlf = QLFHopkins()

    # Initialize figure, redshifts
    fig, ax = plt.subplots(figsize=(4, 3))
    zs = np.linspace(0, 6, 500)
    zs = zs[1:]

    ##############################
    ###    Luminosity bins     ###
    ##############################

    # Define luminosity bins
    color2Lbin = {
        "xkcd:blue": (1e44, 1e45) * u.erg / u.s,
        "xkcd:teal": (1e45, 1e46) * u.erg / u.s,
        "xkcd:green": (1e46, 1e47) * u.erg / u.s,
        "xkcd:orange": (1e47, 1e48) * u.erg / u.s,
        "xkcd:red": (1e48, 1e49) * u.erg / u.s,
    }

    # Iterate over redshifts
    dn_d3Mpc = []
    for color, Lbin in color2Lbin.items():
        # Get number density
        dn_d3Mpc_temp = qlf.dn_d3Mpc(
            zs=zs,
            cosmo=cosmo,
            brightness_limits=Lbin,
        )

        # Append to list
        dn_d3Mpc.append(dn_d3Mpc_temp)

    # Cast to array
    dn_d3Mpc = u.Quantity(dn_d3Mpc)

    # Convert to dn/dOmega/dz
    dn_dOmega_dz = dn_d3Mpc * cosmo.differential_comoving_volume(zs)

    # Plot the number density as a function of redshift
    for ci, color in enumerate(color2Lbin.keys()):
        ax.plot(
            zs,
            np.log10(dn_dOmega_dz[ci, :].value),
            color=color,
            lw=2,
        )

    # Check min and max values
    assert dn_dOmega_dz.min().to(u.sr**-1).value == pytest.approx(
        0.0009601991928953038, rel=1e-7
    )
    assert dn_dOmega_dz.max().to(u.sr**-1).value == pytest.approx(
        5113430.529626767, rel=1000
    )

    ##############################
    ###       g<20.5 mag       ###
    ##############################

    # Get number density
    dn_dOmega_dz = qlf.dn_dOmega_dz(
        zs=zs,
        cosmo=cosmo,
        brightness_limits=(20.5, -np.inf) * u.ABmag,
    )

    # Plot
    ax.plot(
        zs,
        np.log10(dn_dOmega_dz.to(u.sr**-1).value),
        color="xkcd:black",
        linestyle="--",
    )

    # Check min and max values
    assert dn_dOmega_dz.min().to(u.sr**-1).value == pytest.approx(
        10.043361526747251, rel=1e-2
    )
    assert dn_dOmega_dz.max().to(u.sr**-1).value == pytest.approx(
        190765.70313124405, rel=100
    )

    ##############################
    ###     Clean and save     ###
    ##############################

    # Set bounds, add labels
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7.5)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$\log_{10} \left( dn / (d\Omega dz)~[{\rm sr}^{-1}] \right)$")

    # Save figure
    plt.tight_layout()
    figpath = (
        f"{pa.dirname(__file__)}/figures/{pa.basename(__file__).replace('.py', '.png')}"
    )
    os.makedirs(pa.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, dpi=300)
    plt.close()
    assert True


if __name__ == "__main__":
    test_palmese21()
