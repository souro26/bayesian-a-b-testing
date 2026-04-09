import numpy as np


class BaseModel:
    """
    Base class for all Bayesian models.

    Input: dict[str, np.ndarray] where each key is a variant
    Output: posterior samples as np.ndarray of shape (n_draws, n_variants)
    """

    def __init__(self):
        """Initialize empty model state."""
        self.data = None
        self.variant_names = None

    def fit(self, data: dict[str, np.ndarray]) -> None:
        """Validate and store input data."""
        self._validate_input(data)
        self.data = {k: np.asarray(v) for k, v in data.items()}
        self.variant_names = list(self.data.keys())

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """Generate posterior samples. Must be implemented by subclasses."""
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