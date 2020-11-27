"""Automatic Quadrature."""

from ._bayesian_quadrature import BayesianQuadrature


class AutomaticBayesianQuadrature(BayesianQuadrature):
    """
    Automatic Bayesian Quadrature.
    """

    def __init__(self, fun ):

    def integrate(self, fun, domain, measure, nevals):
        """
        Integrate the function ``fun``.

        Parameters
        ----------
        fun :
            Function to be integrated.
        domain :
            Domain to integrate over.
        measure :
            Measure to integrate against.
        nevals :
            Number of function evaluations.

        Returns
        -------

        """
        raise NotImplementedError
