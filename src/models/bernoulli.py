import numpy as np
from scipy.stats import beta as beta_dist


class BernoulliModel:
    """
    Bayesian Bernoulli model using Beta conjugate prior.
    Provides posterior sampling for conversion probability.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Prior parameters must be strictly positive.")

        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)

    def _validate_inputs(self, successes: int, trials: int):
        if trials < 0:
            raise ValueError("Trials must be non-negative.")

        if successes < 0:
            raise ValueError("Successes must be non-negative.")

        if successes > trials:
            raise ValueError("Successes cannot exceed trials.")

    def posterior_parameters(self, successes: int, trials: int):
        """
        Compute posterior alpha and beta parameters.
        """

        self._validate_inputs(successes, trials)

        alpha_post = self.alpha_prior + successes
        beta_post = self.beta_prior + (trials - successes)

        return alpha_post, beta_post

    def sample_posterior(
        self,
        successes: int,
        trials: int,
        n_samples: int = 50_000,
    ):
        """
        Draw posterior samples of conversion probability.
        """

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        alpha_post, beta_post = self.posterior_parameters(successes, trials)

        samples = beta_dist.rvs(alpha_post, beta_post, size=n_samples)

        return samples
