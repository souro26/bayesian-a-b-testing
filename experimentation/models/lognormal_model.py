import numpy as np
import pymc as pm
from .base_model import BaseModel


class LogNormalModel(BaseModel):
    """Bayesian model for positive, right-skewed data using LogNormal."""

    def __init__(self, mu_prior_mean=0.0, mu_prior_sd=10.0, sigma_prior_sd=5.0):
        """Initialize priors for mu and sigma."""
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_sd = mu_prior_sd
        self.sigma_prior_sd = sigma_prior_sd
        super().__init__()

    def _validate_input(self, data) -> None:
        """Ensure inputs are valid and strictly positive."""
        super()._validate_input(data)

        for k, v in data.items():
            if (v <= 0).any():
                raise ValueError(f"{k} must contain only positive values")

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """Return posterior samples of expected value per variant."""
        if self.data is None or self.variant_names is None:
            raise ValueError("Call fit() before sampling")

        samples = []

        for variant in self.variant_names:
            data = self.data[variant]

            with pm.Model():
                mu = pm.Normal("mu", mu=self.mu_prior_mean, sigma=self.mu_prior_sd)
                sigma = pm.HalfNormal("sigma", sigma=self.sigma_prior_sd)

                pm.LogNormal("obs", mu=mu, sigma=sigma, observed=data)

                trace = pm.sample(
                    draws=n_draws,
                    tune=1000,
                    chains=1,  
                    progressbar=False,
                    random_seed=42
                )

            mu_samples = trace.posterior["mu"].values.flatten()
            sigma_samples = trace.posterior["sigma"].values.flatten()

            mean_samples = np.exp(mu_samples + 0.5 * sigma_samples**2)

            samples.append(mean_samples)

        return np.column_stack(samples)