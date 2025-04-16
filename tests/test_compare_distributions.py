import os.path as pa

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from myagn.distributions import ConstantPhysicalDensity, Milliquas, QLFHopkins


def test_compare_distributions():
    """Compare QLFHopkins and Milliquas distributions."""

    # Initialize figure
    fig, ax = plt.subplots(figsize=(6 * 2 / 3, 5 * 2 / 3))
    zs = np.linspace(0, 6, 100)
    zs = zs[1:]

    ##############################
    ###       ConstantPhysicalDensity       ###
    ##############################

    # Initialize distribution
    cpd = ConstantPhysicalDensity((10**-4.75) * u.Mpc**-3)
    ax.plot(
        zs,
        np.log10(cpd.dn_dOmega_dz(zs=zs).value),
        label=r"ConstantPhysicalDensity $10^{-4.75}$",
    )

    ##############################
    ###       QLFHopkins       ###
    ##############################

    # Initialize distribution
    qlf_hopkins = QLFHopkins()

    # Get number density
    dn_dOmega_dz_hopkins = qlf_hopkins.dn_dOmega_dz(
        zs=zs,
        brightness_limits=(20.5, -np.inf) * u.ABmag,
        band="g",
    )

    # Plot
    ax.plot(
        zs,
        np.log10(dn_dOmega_dz_hopkins.value),
        label="QLFHopkins (gmag < 20.5)",
    )

    ##############################
    ###         Milliquas       ###
    ##############################

    # Initialize distribution
    milliquas = Milliquas()

    # Mask to confident AGN
    mask = np.array(["q" not in t for t in milliquas._catalog["Type"]])
    milliquas._catalog = milliquas._catalog[mask]

    # Get number density
    dn_dOmega_dz_milliquas = milliquas.dn_dOmega_dz(
        zs=zs,
    )

    # Plot
    ax.plot(
        zs,
        np.log10(dn_dOmega_dz_milliquas.value),
        label="Milliquas",
    )

    ##############################
    ###     Clean and save     ###
    ##############################

    # Set bounds, add labels
    ax.set_xlim(0, 6)
    # ax.set_ylim(0, 7)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$\log_{10} \left( dn / dz \right)$")

    # Save figure
    plt.legend()  # loc="upper right")
    plt.tight_layout()
    figpath = (
        f"{pa.dirname(__file__)}/figures/{pa.basename(__file__).replace('.py', '.png')}"
    )
    plt.savefig(figpath, dpi=300)
    plt.close()
    assert True


if __name__ == "__main__":
    test_compare_distributions()
