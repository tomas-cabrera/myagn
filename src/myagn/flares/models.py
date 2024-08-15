"""Module handling AGN flare modeling"""

import astropy.units as u
import pandas as pd
from scipy.stats import norm


class AGNFlareModel:
    """A class to handle AGN flare models

    Returns
    -------
    _type_
        _description_
    """

    def __init__(self) -> None:
        pass


class ConstantRate(AGNFlareModel):
    """Class to handle a constant flare rate model"""

    def __init__(self, flare_rate) -> None:
        super().__init__()
        self._flare_rate = flare_rate

    def flare_rate(self, *args, **kwargs):
        return self._flare_rate


class Kimura20(AGNFlareModel):
    """A class to handle flare models a la Kimura+20:
    DOI: 10.3847/1538-4357/ab83f3

    Parameters
    ----------
    AGNFlareModel : _type_
        _description_
    """

    def __init__(self) -> None:
        super().__init__()
        # Structure function parameters from Kimura+20 S2.1,
        self._sf_params = pd.DataFrame(
            [
                {
                    "band": "g",
                    "SF0": 0.210,
                    "SF0err": 0.003,
                    "dt0": 100,
                    "bt": 0.411,
                    "bterr": 0.13,
                },
                {
                    "band": "r",
                    "SF0": 0.160,
                    "SF0err": 0.002,
                    "dt0": 100,
                    "bt": 0.440,
                    "bterr": 0.12,
                },
                {
                    "band": "i",
                    "SF0": 0.133,
                    "SF0err": 0.001,
                    "dt0": 100,
                    "bt": 0.511,
                    "bterr": 0.10,
                },
                {
                    "band": "z",
                    "SF0": 0.097,
                    "SF0err": 0.001,
                    "dt0": 100,
                    "bt": 0.492,
                    "bterr": 0.13,
                },
            ]
        ).set_index("band")

    def structure_function(self, band, dt):
        """Computes the structure function for the given time delta.
        Multiple bands/dts may be passed as array-like objects.

        Parameters
        ----------
        band : _type_
            _description_
        dt : _type_
            The time interval for the given dmag, in rest-frame days.

        Returns
        -------
        _type_
            _description_
        """
        # Get band parameters
        sfps = self._sf_params.loc[band]
        # Calculate structure function
        sf = sfps["SF0"] * (dt / sfps["dt0"]) ** sfps["bt"]
        return sf

    def flare_rate(self, *args, **kwargs):
        """Computes the flare rate for the given parameters.
        Arrays may be passed to all arguments;
        if multiple arrays are passed, then they must be of the same dimensionality.

        Parameters
        ----------
        band : _type_
            _description_
        dt : _type_
            The time interval for the given dmag, in rest-frame days.
        dmag : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Extract parameters
        band, dt, dmag = args
        # Calculate structure function
        sf = self.structure_function(band, dt)
        # Calculate rate
        rate = 1 - norm.cdf(dmag, loc=0, scale=sf)
        return rate
