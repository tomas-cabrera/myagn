"""Module handling AGN distributions"""

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .qlfhopkins import qlfhopkins


class AGNDistribution:
    """A class to handle AGN distributions"""

    def __init__(self) -> None:
        pass

    def n_agn_in_dOmega_dz_volume(
        self,
        dOmega=500,
        dz_grid=np.linspace(0, 1, 50),
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        magnitude_limits=(np.inf, -np.inf),
    ):
        """Calculate number of AGNs in spherical volume element,
        defined by a steradian area dOmega and a redshift grid dz_grid.

        Parameters
        ----------
        dOmega : int, optional
            _description_, by default 500
        dz_grid : _type_, optional
            _description_, by default np.linspace(0, 1, 50)
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        magnitude_limits : _type_, optional
            _description_, by default (np.inf, -np.inf)

        Returns
        -------
        _type_
            _description_
        """

        # Calculate number densities at redshift grids
        dn_dOmega_dz = self.dn_dOmega_dz(
            zs=dz_grid,
            cosmo=cosmo,
            magnitude_limits=magnitude_limits,
        )

        # Sum, multiply by elements to get total number of agns
        n_agns = np.sum(dn_dOmega_dz) * dOmega * (dz_grid[1] - dz_grid[0])

        return n_agns


class QLFHopkins(AGNDistribution):
    """AGN distribution modeled after Hopkins+06

    Parameters
    ----------
    AGNDistribution : _type_
        _description_
    """

    def dn_dOmega_dz(
        self,
        zs=np.linspace(0, 1, 50),
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        magnitude_limits=(np.inf, -np.inf),
    ):
        """Calculate number density of visible AGNs at given redshifts.

        Parameters
        ----------
        zs : _type_, optional
            _description_, by default np.linspace(0, 1, 50)
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        magnitude_limits : _type_, optional
            _description_, by default (np.inf, -np.inf)

        Returns
        -------
        _type_
            _description_
        """
        # Calculate limiting flux
        f_nu_det = (magnitude_limits * u.ABmag).to(u.erg / u.s / u.cm**2 / u.Hz)
        nu = (const.c / (475 * u.nm)).to(u.Hz)  # ZTF g-band
        f_det = f_nu_det * nu

        dn_dOmega_dz_visible = []
        for z in zs:
            # Calculate limiting luminosity, assuming bolometric luminosity is 10x g-band
            L_bol_det = (10 * f_det * 4 * np.pi * cosmo.luminosity_distance(z) ** 2).to(
                u.erg / u.s
            )

            # Get qlf info
            df_qlf = qlfhopkins.compute(0, z)

            # Sum all number densities for luminosities larger than limit
            dn_dlog10L_d3Mpc_visible = df_qlf[
                (df_qlf["bolometric_luminosity"] >= np.log10(L_bol_det[0].value))
                & (df_qlf["bolometric_luminosity"] < np.log10(L_bol_det[1].value))
            ]["comoving_number_density"].sum()

            # Convert to dOmega_dz volume and append
            # dlog10L = 0.1 (standard for qlfhopkins)
            dn_dOmega_dz_visible.append(
                dn_dlog10L_d3Mpc_visible * 0.1 * cosmo.differential_comoving_volume(z)
            )

        return dn_dOmega_dz_visible
