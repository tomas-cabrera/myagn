"""Module handling AGN distributions"""

import astropy.constants as const
import astropy.cosmology.units as cu
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
        n_agns = np.trapezoid(dn_dOmega_dz, dz_grid) * dOmega

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
        brightness_limits=None,
    ):
        """Calculate number density of visible AGNs at given redshifts.

        Parameters
        ----------
        zs : _type_, optional
            _description_, by default np.linspace(0, 1, 50)
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """

        # Get density of AGNs in Mpc^-3
        dn_d3Mpc = self.dn_d3Mpc(
            zs=zs,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Convert to dn/dOmega/dz
        dn_dOmega_dz = dn_d3Mpc * cosmo.differential_comoving_volume(zs)

        return dn_dOmega_dz

    def dn_d3Mpc(
        self,
        zs=np.linspace(0, 1, 50),
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
    ):
        """Calculate number density of visible AGNs at given redshifts.

        Parameters
        ----------
        zs : _type_, optional
            _description_, by default np.linspace(0, 1, 50)
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # Do initial limit calculations
        luminosity_limits = None
        flux_limits = None
        if brightness_limits is None:
            luminosity_limits = (-np.inf, np.inf) * u.erg / u.s
        # If a magnitude or flux is used
        elif brightness_limits.unit.is_equivalent(u.ABmag):
            # Assume ZTF g-band magnitude/flux for now
            f_nu_det = (brightness_limits).to(u.erg / u.s / u.cm**2 / u.Hz)
            nu = (const.c / (475 * u.nm)).to(u.Hz)
            flux_limits = f_nu_det * nu
        # If a luminosity is used
        elif brightness_limits.unit.is_equivalent(u.erg / u.s):
            luminosity_limits = brightness_limits
        else:
            raise NotImplementedError(
                "brightness_limits must have units of magnitude, flux, or luminosity"
            )

        # Iterate over redshifts
        dn_d3Mpc = []
        for z in zs:
            # Calculate luminosity limits if needed;
            # calculation is triggered if flux_limits is not None
            if flux_limits is not None:
                # Calculate limiting luminosity, assuming bolometric luminosity is 10x g-band
                luminosity_limits = (
                    10 * flux_limits * 4 * np.pi * cosmo.luminosity_distance(z) ** 2
                ).to(u.erg / u.s)

            # Get qlf info
            df_qlf = qlfhopkins.compute(0, z)

            # Sum all number densities for luminosities larger than limit
            dn_dlog10L_d3Mpc = df_qlf[
                (
                    df_qlf["bolometric_luminosity"]
                    >= np.log10(luminosity_limits[0].value)
                )
                & (
                    df_qlf["bolometric_luminosity"]
                    < np.log10(luminosity_limits[1].value)
                )
            ]["comoving_number_density"].sum()

            # Append to list (0.1 is the luminosity bin width)
            dn_d3Mpc.append(dn_dlog10L_d3Mpc * 0.1)

        # Convert to numpy array, add units
        dn_d3Mpc = np.array(dn_d3Mpc) * u.Mpc**-3

        return dn_d3Mpc
