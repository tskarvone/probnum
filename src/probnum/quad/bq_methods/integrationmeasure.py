""""
Integration measure class for Bayesian quadrature.

These classes implement various integration measures used in Bayesian quadrature.

List of measures:
-----------------
IntegrationMeasure
    Generic measure class which can be provided with a custom density function.
LebesgueMeasure
    Lebesgue (i.e., uniform) measure on a given hyper-rectangle. Can be either
    normalised or unnormalised.
GaussianMeasure
    Gaussian measure (i.e., normal distribution) on the whole multi-dimensional real
    space.
"""
import abc
from typing import Callable, Optional, Union

import numpy as np

from probnum.type import IntArgType, FloatArgType


class IntegrationMeasure(abc.ABC):
    """
    Generic class for integration measures.
    """

    def __init__(self,
                 ndim: Optional[IntArgType] = 1,
                 a: Optional[FloatArgType] = 0.,
                 b: Optional[FloatArgType] = 1.,
                 density_fun: Optional[Callable[[Union[np.ndarray, FloatArgType]],
                                                FloatArgType]] = None
                 ):

        self._set_integration_limits(ndim, a, b)
        self.density_fun = density_fun

    def evaluate_density(self,
                         points: np.ndarray
                         ) -> np.ndarray:
        if self.density_fun is None:
            raise TypeError("No density function has been defined.")
        npoints = len(points)
        density_evals = np.zeros(npoints)
        for idx in range(npoints):
            density_evals[idx] = self.density_fun(points[idx])
        return density_evals

    def _set_integration_limits(self, ndim, a, b):
        """
        Check that the provided integration limits ``a`` and ``b`` which define the
        hyper-rectangular integration domain
            [ a[0], b[0] ] x ... x [ a[ndim-1], b[ndim-1] ]
        are consistent with one another and the dimension ``ndim``. If the dimension
        exceeds one and either (or both) of the limit arrays is a scalar, this array is
        expanded as a constant-value array to produce, for example, a domain
            [ a, b[0] ] x ... x [ a, b[ndim-1] ] .
        """
        # Check dimensions
        if (len(a) > 1 and len(b) > 1) and (len(a) != len(b) or len(a) != ndim
                                            or len(b) != ndim):
            raise ValueError(
                f"Integration limit arrays have inconsistent lengths: Dimension is "
                f"{ndim} but a has length {a.shape[0]} and b has length {b.shape[0]}."
            )
        if ndim > 1 and len(a) == 1:
            a = np.full(ndim, a)
        if ndim > 1 and len(b) == 1:
            b = np.full(ndim, b)

        # Make sure intervals are properly defined
        if not all(a < b):
            raise ValueError(
                "There are integration limit start points that are larger than "
                "corresponding end points."
            )

        self.a = a
        self.b = b
        self.ndim = ndim


class LebesgueMeasure(IntegrationMeasure):
    """
    Lebesgue (i.e., uniform) measures.
    """

    def __init__(self, ndim=1, a=0., b=1., normalized=True):
        super().__init__(ndim=ndim, a=a, b=b)
        self._set_normalization_constant(normalized)

    def evaluate_density(self, points):
        return np.full(len(points), self.normalization_constant)

    def _set_normalization_constant(self, normalized):
        if normalized:
            try:
                normalization_constant = np.prod(1./(self.b-self.a))
            except ZeroDivisionError as e:
                raise Exception(
                    f"Definition of the normalization constant of {type(self)} "
                    "resulted in division by zero. Make sure that integration limits "
                    "yield non-empty intervals."
                ) from e
        else:
            normalization_constant = 1.

        if np.isnan(normalization_constant):
            raise ValueError(f"Normalization constant for {type(self)} is not defined. "
                             "Make sure that integration limits are finite.")

        if normalization_constant < 0:
            raise ValueError(f"Normalization constant for {type(self)} is negative "
                             "Make sure that integration limits are correctly defined.")

        self.normalization_constant = normalization_constant


class GaussianMeasure(IntegrationMeasure):
    """
    Gaussian (i.e., normal) measures.
    """

    def __init__(self, ndim=1, measure_mean=0, measure_var=1):
        raise NotImplementedError
        #super().__init__(ndim=ndim, a=-np.inf, b=np.inf)


