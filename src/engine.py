import numpy as np
from typing import Dict

from src.models.bernoulli_model import BernoulliModel
from src.models.lognormal_model import LogNormalModel
from src.decision.decision_engine import decision_engine


class BayesianABTest:
    """
    High-level orchestrator for Bayesian A/B testing.

    Supports:
    - Conversion-only testing
    - Revenue per user testing (conversion × revenue per conversion)

    This class coordinates models and decision logic.
    """

    def __init__(self):
        self.conversion_model = BernoulliModel()
        self.revenue_model = LogNormalModel()

    def run_test(
        self,
        variant_a: Dict,
        variant_b: Dict,
        metric: str = "conversion",
        rope: float = 0.01,
        loss_threshold: float = 0.01,
        n_samples: int = 50_000,
    ):
        """
        Run Bayesian A/B test for specified metric.

        Parameters
        ----------
        variant_a : dict
            {
                "successes": int,
                "trials": int,
                "revenue": np.ndarray (optional, required for revenue metric)
            }

        variant_b : dict
            Same structure as variant_a.

        metric : str
            "conversion" or "revenue_per_user"

        rope : float
            Region of Practical Equivalence threshold.

        loss_threshold : float
            Maximum acceptable expected loss.

        n_samples : int
            Number of Monte Carlo posterior samples.
        """

        if metric not in {"conversion", "revenue_per_user"}:
            raise ValueError("metric must be 'conversion' or 'revenue_per_user'.")

        # --- Conversion samples ---
        conv_samples_a = self.conversion_model.sample_posterior(
            variant_a["successes"],
            variant_a["trials"],
            n_samples,
        )

        conv_samples_b = self.conversion_model.sample_posterior(
            variant_b["successes"],
            variant_b["trials"],
            n_samples,
        )

        if metric == "conversion":
            samples_a = conv_samples_a
            samples_b = conv_samples_b

        elif metric == "revenue_per_user":

            if "revenue" not in variant_a or "revenue" not in variant_b:
                raise ValueError(
                    "Revenue arrays must be provided for revenue_per_user metric."
                )

            revenue_a = np.asarray(variant_a["revenue"])
            revenue_b = np.asarray(variant_b["revenue"])

            # Revenue per conversion samples
            rev_samples_a = self.revenue_model.sample_predictive(
                revenue_a,
                n_samples,
            )

            rev_samples_b = self.revenue_model.sample_predictive(
                revenue_b,
                n_samples,
            )

            # Revenue per user = conversion × revenue
            samples_a = conv_samples_a * rev_samples_a
            samples_b = conv_samples_b * rev_samples_b

        # --- Decision ---
        return decision_engine(
            samples_a,
            samples_b,
            rope=rope,
            loss_threshold=loss_threshold,
        )
