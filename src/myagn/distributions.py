"""Module handling AGN distributions"""

import astropy.constants as const
import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from myagn.qlfhopkins import qlfhopkins


class AGNDistribution:
    """A class to handle AGN distributions"""

    def __init__(self) -> None:
        pass

    def dn_dOmega_dz(
        self,
        zs,
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
            Tuple-like of lower and upper brightness limits.
            If None (default), then no brightness cut is applied.
            The units of the argument are used to determine usage:
                - if magnitude or flux, then assumed to be the limits for ZTF g-band
                - if luminosity, then assumed to be bolometric luminosity

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

    def n_agn_in_DOmega_Dz_slice(
        self,
        z_grid,
        DOmega=4 * np.pi * u.sr,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=(np.inf, -np.inf),
    ):
        """Calculate number of AGNs in spherical volume element,
        defined by a steradian area dOmega and a redshift grid dz_grid.

        Parameters
        ----------
        dOmega : int, optional
            _description_, by default 500
        z_grid : _type_, optional
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
            zs=z_grid,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Sum, multiply by elements to get total number of agns
        n_agn = np.trapezoid(dn_dOmega_dz, z_grid) * DOmega.to(u.sr)

        return n_agn

    def dp_dOmega_dz(
        self,
        z_grid,
        z_evaluate=None,
        DOmega=4 * np.pi * u.sr,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
    ):
        """Convert dn_dOmega_dz to probability density.
        This is done by integrating over redshift and solid angle.

        Parameters
        ----------
        z_grid : _type_, optional
            The redshift domain for the probability density, by default np.linspace(0, 1, 50)*cu.redshift
        z_evaluate : _type_, optional
            The redshift array to evaluate the probability density at.
            By default None; if None, then z_evaluate=z_grid (the probability density is evaluated at every point in z_grid)
        DOmega : _type_, optional
            _description_, by default 4*np.pi*u.sr
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # If no z_evaluate is given, use z_grid
        if z_evaluate is None:
            z_evaluate = z_grid

        # Get number distribution
        dn_dOmega_dz = self.dn_dOmega_dz(
            zs=z_evaluate,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Get number in grid
        n_agn = self.n_agn_in_DOmega_Dz_slice(
            DOmega=DOmega,
            z_grid=z_grid,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Normalize
        dp_dOmega_dz = dn_dOmega_dz / n_agn

        return dp_dOmega_dz

    def sample_z(
        self,
        n_samples,
        z_grid,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
        rng_np=np.random.default_rng(12345),
    ):
        """Sample redshifts from the probability density function.

        Parameters
        ----------
        n_samples : _type_
            _description_
        z_grid : _type_
            _description_
        DOmega : _type_, optional
            _description_, by default 4*np.pi*u.sr
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # Get probability density
        dp_dOmega_dz = self.dn_dOmega_dz(
            z_grid=z_grid,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Sample redshifts
        z_samples = rng_np.choice(z_grid, size=n_samples, p=dp_dOmega_dz)

        return z_samples


class ConstantPhysicalDensity(AGNDistribution):
    """AGN distribution with constant physical number density"""

    def __init__(self, n_per_Mpc3) -> None:
        """_summary_

        Parameters
        ----------
        n_per_Mpc3 : _type_
            The number density of AGNs to use
        """
        super().__init__()
        self.n_per_Mpc3 = n_per_Mpc3

    def dn_d3Mpc(
        self,
        zs,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
    ):
        return u.quantity(self.n_per_Mpc3.to(u.Mpc**-3) * np.ones_like(zs))


class QLFHopkins(AGNDistribution):
    """AGN distribution modeled after Hopkins+06

    Parameters
    ----------
    AGNDistribution : _type_
        _description_
    """

    def dn_d3Mpc(
        self,
        zs,
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

            # Append to list
            # (0.1 is the luminosity bin width used in the qlfhopkins calculator,
            #   so we multiply to integrate over log10L)
            dn_d3Mpc.append(dn_dlog10L_d3Mpc * 0.1)

        # Convert to numpy array, add units
        dn_d3Mpc = np.array(dn_d3Mpc) * u.Mpc**-3

        return dn_d3Mpc
