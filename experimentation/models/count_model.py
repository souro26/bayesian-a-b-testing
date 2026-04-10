import numpy as np
import pymc as pm
from .base_model import BaseModel


class CountModel(BaseModel):
    """Bayesian model for count data using Poisson likelihood."""

    def __init__(self, lam_prior_alpha=1.0, lam_prior_beta=1.0):
        """Initialize Gamma prior parameters for rate λ."""
        super().__init__()
        self.lam_prior_alpha = lam_prior_alpha
        self.lam_prior_beta = lam_prior_beta

    def _validate_input(self, data) -> None:
        """Ensure inputs are non-negative integers."""
        super()._validate_input(data)

        for k, v in data.items():
            if (v < 0).any():
                raise ValueError(f"{k} must contain non-negative values")

            if not np.issubdtype(v.dtype, np.integer):
                raise ValueError(f"{k} must contain integer values")

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """Return posterior samples of the Poisson rate λ for each variant."""
        if self.data is None or self.variant_names is None:
            raise ValueError("Call fit() before sampling")
        
        if n_draws == 0:
            return np.empty((0, len(self.variant_names)))

        samples = []

        for variant in self.variant_names:
            data = self.data[variant]

            with pm.Model():
                lam = pm.Gamma("lam", alpha=self.lam_prior_alpha, beta=self.lam_prior_beta)
                
                pm.Poisson("obs", mu=lam, observed=data)

                trace = pm.sample(
                    draws=n_draws,
                    tune=min(1000, n_draws),
                    chains=1,
                    progressbar=False,
                    random_seed=42
                )

            lam_samples = trace.posterior["lam"].values.flatten()
            samples.append(lam_samples)

        return np.column_stack(samples)