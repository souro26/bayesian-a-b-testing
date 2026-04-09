import numpy as np
import pymc as pm
from .base_model import BaseModel


class GaussianModel(BaseModel):
    """Bayesian model for normally distributed data."""

    def __init__(self, mu_prior_mean=0.0, mu_prior_sd=10.0, sigma_prior_sd=5.0):
        """Initialize priors for mu and sigma."""
        super().__init__()
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_sd = mu_prior_sd
        self.sigma_prior_sd = sigma_prior_sd

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """Return posterior samples of the mean for each variant."""
        if self.data is None or self.variant_names is None:
            raise ValueError("Call fit() before sampling")

        samples = []

        for variant in self.variant_names:
            data = self.data[variant]

            with pm.Model():
                mu = pm.Normal("mu", mu=self.mu_prior_mean, sigma=self.mu_prior_sd)
                sigma = pm.HalfNormal("sigma", sigma=self.sigma_prior_sd)

                pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

                trace = pm.sample(
                    draws=n_draws,
                    tune=1000,
                    chains=1,
                    progressbar=False,
                    random_seed=42
                )

            mu_samples = trace.posterior["mu"].values.flatten()
            samples.append(mu_samples)

        return np.column_stack(samples)
    

class StudentTModel(BaseModel):
    """Bayesian model for heavy-tailed data using Student-t distribution."""

    def __init__(self, mu_prior_mean=0.0, mu_prior_sd=10.0, sigma_prior_sd=5.0):
        """Initialize priors and fixed degrees of freedom."""
        super().__init__()
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_sd = mu_prior_sd
        self.sigma_prior_sd = sigma_prior_sd

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """Return posterior samples of the mean for each variant."""
        if self.data is None or self.variant_names is None:
            raise ValueError("Call fit() before sampling")

        samples = []

        for variant in self.variant_names:
            data = self.data[variant]

            with pm.Model():
                mu = pm.Normal("mu", mu=self.mu_prior_mean, sigma=self.mu_prior_sd)
                sigma = pm.HalfNormal("sigma", sigma=self.sigma_prior_sd)
                nu = pm.Exponential("nu", lam=1/30)
                
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=data)

                trace = pm.sample(
                    draws=n_draws,
                    tune=1000,
                    chains=1,
                    progressbar=False,
                    random_seed=42
                )

            mu_samples = trace.posterior["mu"].values.flatten()
            samples.append(mu_samples)

        return np.column_stack(samples)