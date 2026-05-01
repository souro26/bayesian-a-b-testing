import numpy as np


class BaseModel:
    """
    Abstract base architecture for Bayesian models in the decision-making engine.

    Models inheriting from this class are expected to consume dictionaries mapping
    variant identifiers to raw observed metric arrays, and subsequently generate rigorous
    posterior distributions using Monte Carlo methods (typically via PyMC). The output 
    samples must conform uniformly to an `(n_draws, n_variants)` geometric shape so that 
    the decision engine can transparently operate over different generative models.

    Attributes
    ----------
    data : dict[str, np.ndarray] | None
        The ingested observation data organized by variant.
    variant_names : list[str] | None
        A sorted reference mapping columns of output arrays to specific variants.
    """

    def __init__(self):
        """Initialize empty model state."""
        self.data = None
        self.variant_names = None

    def fit(self, data: dict) -> None:
        """
        Validate, sort, and permanently attach the observational data to the model.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            A payload mapping qualitative variant designators to arrays of discrete 
            or continuous numerical observations.

        Returns
        -------
        None
        """
        self._validate_input(data)
        self.data = {k: np.array(v) for k, v in data.items()}
        self.variant_names = sorted(data.keys())

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """
        Generate empirical posterior samples from the underlying probabilistc graph.

        Parameters
        ----------
        n_draws : int, optional
            The total number of MCMC draws requested from the sampler, by default 2000.

        Returns
        -------
        np.ndarray
            A strictly formatted 2D array of localized posterior draws (`n_draws` rows 
            by `len(variant_names)` columns).

        Raises
        ------
        NotImplementedError
            Always raised if called directly; must be implemented by subclasses.
        """
        raise NotImplementedError

    def _validate_input(self, data) -> None:
        """Validate structure and basic integrity of input data."""
        if not isinstance(data, dict):
            raise TypeError("Data must be dict[str, np.ndarray]")

        if len(data) < 2:
            raise ValueError("At least two variants required")

        for k, v in data.items():
            if not isinstance(v, np.ndarray):
                raise TypeError(f"{k} must be numpy array")

            if len(v) == 0:
                raise ValueError(f"{k} has no data")

            if v.ndim != 1:
                raise ValueError(f"{k} must be 1D array")

            if np.isnan(v).any():
                raise ValueError(f"{k} contains NaNs")
            

# TODO: expose n_draws and tune as user-configurable parameters
# TODO: add fast_sample() method with reduced draws for development use
# TODO: mock pm.sample in unit tests for speed