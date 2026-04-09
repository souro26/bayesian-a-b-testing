import numpy as np
from .base_model import BaseModel


class BinaryModel(BaseModel):
    """Bayesian model for binary outcomes."""

    def _validate_input(self, data) -> None:
        """Ensure inputs are valid and binary."""
        super()._validate_input(data)

        for k, v in data.items():
            if not np.isin(v, [0, 1]).all():
                raise ValueError(f"{k} must contain only binary (0/1) values")

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """Return posterior samples using Beta-Bernoulli update."""
        if self.data is None or self.variant_names is None:
            raise ValueError("Call fit() before sampling")

        samples = []

        for variant in self.variant_names:
            data = self.data[variant]

            successes = np.sum(data)
            failures = data.shape[0] - successes

            draws = np.random.beta(
                1 + successes,
                1 + failures,
                size=n_draws
            )

            samples.append(draws)

        return np.column_stack(samples)