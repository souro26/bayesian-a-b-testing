import numpy as np
import pymc as pm
from .base_model import BaseModel


class LogNormalModel(BaseModel):
    """
    Bayesian model for log-normal distributed outcomes.

    Use case: Revenue, order value, time-on-page — any positive, right-skewed metric.

    Model: log(data) ~ Normal(mu, sigma)
    Returns posterior samples of the expected value (mean) for each variant.
    """

    def __init__(self, mu_prior_mean: float = 0.0, mu_prior_sd: float = 10.0,
                 sigma_prior_sd: float = 5.0):
        """
        Initialize LogNormal model with prior parameters.

        Args:
            mu_prior_mean: Mean of Normal prior on mu (log-scale mean)
            mu_prior_sd: SD of Normal prior on mu
            sigma_prior_sd: SD of HalfNormal prior on sigma (log-scale SD)
        """
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_sd = mu_prior_sd
        self.sigma_prior_sd = sigma_prior_sd
        super().__init__()

    def _validate_input(self, data) -> None:
        """Ensure inputs are valid and strictly positive."""
        super()._validate_input(data)

        for k, v in data.items():
            if (v <= 0).any():
                raise ValueError(
                    f"{k} must contain strictly positive values (> 0). "
                    f"LogNormal cannot model zero or negative values."
                )

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """
        Return posterior samples of the mean for each variant.

        Uses PyMC to sample from the joint posterior of (mu, sigma), then
        transforms to the mean: E[X] = exp(mu + sigma^2 / 2)

        Returns:
            np.ndarray of shape (n_draws, n_variants)
        """
        if self.data is None or self.variant_names is None:
            raise ValueError("Call fit() before sampling")

        samples = []

        for variant in self.variant_names:
            data = self.data[variant]

            with pm.Model() as model:
                # Priors on log-scale parameters
                mu = pm.Normal(
                    "mu",
                    mu=self.mu_prior_mean,
                    sigma=self.mu_prior_sd
                )
                sigma = pm.HalfNormal(
                    "sigma",
                    sigma=self.sigma_prior_sd
                )

                # LogNormal likelihood
                pm.LogNormal(
                    "obs",
                    mu=mu,
                    sigma=sigma,
                    observed=data
                )

                # Sample from posterior
                trace = pm.sample(
                    draws=n_draws,
                    tune=1000,
                    chains=2,
                    progressbar=False,
                    random_seed=42
                )

            # Extract posterior samples
            mu_samples = trace.posterior["mu"].values.flatten()
            sigma_samples = trace.posterior["sigma"].values.flatten()

            # Transform to mean: E[X] = exp(mu + sigma^2 / 2)
            mean_samples = np.exp(mu_samples + sigma_samples ** 2 / 2)
            samples.append(mean_samples)

        return np.column_stack(samples)
