import numpy as np


class LogNormalModel:
    """
    Plug-in LogNormal model for positive revenue data.

    Assumes log(revenue) is approximately Normal.
    """

    def estimate_parameters(self, revenue: np.ndarray):
        """
        Estimate mu and sigma from log-transformed revenue data.
        """

        revenue = np.asarray(revenue)

        if revenue.ndim != 1:
            raise ValueError("Revenue must be a 1D array.")

        if len(revenue) < 2:
            raise ValueError("At least two revenue observations are required.")

        if np.any(revenue <= 0):
            raise ValueError("Revenue values must be strictly positive.")

        log_revenue = np.log(revenue)

        mu_hat = np.mean(log_revenue)
        sigma_hat = np.std(log_revenue, ddof=1)  # sample std

        return mu_hat, sigma_hat

    def sample_predictive(self, revenue: np.ndarray, n_samples: int = 50_000):
        """
        Generate posterior predictive samples of revenue per conversion.
        """

        mu_hat, sigma_hat = self.estimate_parameters(revenue)

        normal_samples = np.random.normal(
            loc=mu_hat,
            scale=sigma_hat,
            size=n_samples,
        )

        return np.exp(normal_samples)
