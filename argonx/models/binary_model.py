import numpy as np
import pymc as pm
import arviz as az
import warnings
from .base_model import BaseModel


class BinaryModel(BaseModel):
    """
    Bayesian model for binary outcomes.

    Assumes a Beta-Bernoulli conjugate model for independent binary outcomes.
    """

    def _validate_input(self, data) -> None:
        """Ensure inputs are valid and binary."""
        super()._validate_input(data)

        for k, v in data.items():
            if not np.isin(v, [0, 1]).all():
                raise ValueError(f"{k} must contain only binary (0/1) values")

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """
        Return posterior samples using a fully conjugated Beta-Bernoulli update.

        For each variant, assumes a uniform Beta(1, 1) prior and updates it via the exact 
        analytical conjugate posterior Beta(1 + successes, 1 + failures). This allows for 
        extremely rapid sampling without Hamiltonial Monte Carlo.

        Parameters
        ----------
        n_draws : int, optional
            Number of posterior draws per variant, by default 2000.

        Returns
        -------
        np.ndarray
            2D array of localized posterior draws (`n_draws` rows by variants columns).
        """
        if self.variant_names is None:
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


class HierarchicalBinaryModel(BaseModel):
    """
    Hierarchical Bayesian model for binary outcomes across segments.

    Uses a Beta-Binomial model with partial pooling across segments.
    Segment conversion rates are drawn from a shared Beta hyperprior,
    allowing thin segments to borrow strength from data-rich segments.

    Parameters
    ----------
    prior_alpha : float, optional
        Alpha parameter for the global Beta hyperprior, by default 1.0.
    prior_beta : float, optional
        Beta parameter for the global Beta hyperprior, by default 1.0.
    kappa_prior_beta : float, optional
        Beta parameter for the HalfCauchy prior on concentration, by default 5.0.
    priors : dict | None, optional
        Dictionary to override default priors, by default None.
    """

    MIN_SEGMENT_SIZE = 10

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        kappa_prior_beta: float = 5.0,
        priors: dict | None = None,
    ):
        
        super().__init__()
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.kappa_prior_beta = kappa_prior_beta

        if priors:
            _allowed = {'prior_alpha', 'prior_beta', 'kappa_prior_beta'}
            unknown = set(priors) - _allowed
            if unknown:
                raise ValueError(f"Unknown prior keys: {unknown}. Allowed: {_allowed}")
            self.prior_alpha = priors.get('prior_alpha', self.prior_alpha)
            self.prior_beta = priors.get('prior_beta', self.prior_beta)
            self.kappa_prior_beta = priors.get('kappa_prior_beta', self.kappa_prior_beta)
            
        self._segment_names: list[str] = []
        self._variant_names_hier: list[str] = []
        self._segment_data: dict[str, dict[str, np.ndarray]] = {}
        self._use_noncentered: bool = True
        self._segment_warnings: dict[str, list[str]] = {}
        self._health_warnings: list[str] = []
        self._trace = None

    def fit(self, data: dict[str, dict[str, np.ndarray]]) -> None:
        """
        Fit hierarchical model to segmented data.

        Parameters
        ----------
        data : dict[str, dict[str, np.ndarray]]
            Nested dictionary mapping segment names to variant data arrays.
        """
        self._validate_hierarchical_input(data)

        self._segment_data = {
            seg: {v: np.array(arr) for v, arr in variants.items()}
            for seg, variants in data.items()
        }
        self._segment_names = sorted(data.keys())
        self._variant_names_hier = sorted(next(iter(data.values())).keys())
        self.variant_names = self._variant_names_hier
        self._segment_warnings = {seg: [] for seg in self._segment_names}
        self._health_warnings = []
        self._preflight_checks()

        min_obs = min(
            len(self._segment_data[seg][var])
            for seg in self._segment_names
            for var in self._variant_names_hier
        )
        self._use_noncentered = min_obs < 100

    def _validate_hierarchical_input(self, data) -> None:
        """Validate nested dict structure and binary values."""
        if not isinstance(data, dict):
            raise TypeError(
                "Hierarchical model expects dict[str, dict[str, np.ndarray]]. "
                f"Got {type(data).__name__}."
            )

        if len(data) < 2:
            raise ValueError("At least two segments required for hierarchical model.")

        variant_sets = []
        for seg, variants in data.items():
            if not isinstance(variants, dict):
                raise TypeError(
                    f"Segment '{seg}' must map to dict[str, np.ndarray]. "
                    f"Got {type(variants).__name__}."
                )
            if len(variants) < 2:
                raise ValueError(
                    f"Segment '{seg}' must have at least two variants."
                )
            variant_sets.append(frozenset(variants.keys()))

            for var, arr in variants.items():
                arr = np.asarray(arr)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Segment '{seg}', variant '{var}': expected 1D array, "
                        f"got shape {arr.shape}."
                    )
                if len(arr) == 0:
                    raise ValueError(
                        f"Segment '{seg}', variant '{var}' has no observations."
                    )
                if np.isnan(arr).any():
                    raise ValueError(
                        f"Segment '{seg}', variant '{var}' contains NaN values."
                    )
                if not np.isin(arr, [0, 1]).all():
                    raise ValueError(
                        f"Segment '{seg}', variant '{var}' must be binary (0/1). "
                        f"Found non-binary values."
                    )

        if len(set(variant_sets)) > 1:
            raise ValueError(
                "All segments must have identical variant sets. "
                f"Found inconsistent variants across segments."
            )

    def _preflight_checks(self) -> None:
        """Warn on thin segments before sampling begins."""
        for seg in self._segment_names:
            total_obs = sum(
                len(self._segment_data[seg][var])
                for var in self._variant_names_hier
            )
            min_obs = min(
                len(self._segment_data[seg][var])
                for var in self._variant_names_hier
            )
            if min_obs < self.MIN_SEGMENT_SIZE:
                self._segment_warnings[seg].append(
                    f"Segment '{seg}' has only {min_obs} observations in its "
                    f"smallest variant cell (total across variants: {total_obs}). "
                    f"Posterior will be prior-dominated. Partial pooling will borrow "
                    f"heavily from other segments. Treat estimates with caution."
                )
                warnings.warn(
                    f"Segment '{seg}': only {min_obs} observations. "
                    f"Posterior will be prior-dominated.",
                    UserWarning,
                    stacklevel=3,
                )

    def sample_posterior(self, n_draws: int = 1000) -> np.ndarray:
        """
        Return population-level posterior samples marginalized and grouped by segment.

        Uses either centered or non-centered parameterization depending on cell sparsity
        (triggering non-centered when any subset falls below 100 observations). Returns
        global posterior inferences weighted appropriately by the segment sizes.

        Parameters
        ----------
        n_draws : int, optional
            Number of draws explicitly returned after inference completes. Defaults to 1000.

        Returns
        -------
        np.ndarray
            2D array representing overall posterior inference populated across variants.
        """
        if not self._segment_names:
            raise ValueError("Call fit() before sampling.")

        self._trace = self._run_mcmc(n_draws=n_draws)
        self._run_health_checks(self._trace)
        by_seg = self._extract_by_segment(self._trace, n_draws)

        segment_sizes = np.array([
            sum(len(self._segment_data[seg][var]) for var in self._variant_names_hier)
            for seg in self._segment_names
        ], dtype=float)
        weights = segment_sizes / segment_sizes.sum()

        population = np.einsum('dsv,s->dv', by_seg, weights)
        return population

    def sample_posterior_by_segment(self, n_draws: int = 1000) -> np.ndarray:
        """
        Return pure per-segment localized posterior samples mapped dimensionally.

        If `sample_posterior()` has already been called, reuses the existing cached traces 
        without performing repetitive HMC integration. Otherwise naturally fits the model first.

        Parameters
        ----------
        n_draws : int, optional
            Requested sample count extracted securely from the generated inference traces. 
            Defaults to 1000.

        Returns
        -------
        np.ndarray
            3D tensor structured as `(n_draws, n_segments, n_variants)`.
        """
        if self._trace is None:
            self._trace = self._run_mcmc(n_draws=n_draws)
            self._run_health_checks(self._trace)

        return self._extract_by_segment(self._trace, n_draws)

    def _run_mcmc(self, n_draws: int) -> az.InferenceData:
        """
        Build and sample the joint hierarchical PyMC model.

        Uses non-centered parameterization (logistic-Normal) when any
        segment has fewer than 100 observations per variant cell.
        Uses centered Beta parameterization otherwise.
        """
        n_seg = len(self._segment_names)
        n_var = len(self._variant_names_hier)

        successes = np.zeros((n_seg, n_var), dtype=int)
        totals = np.zeros((n_seg, n_var), dtype=int)

        for i, seg in enumerate(self._segment_names):
            for j, var in enumerate(self._variant_names_hier):
                arr = self._segment_data[seg][var]
                successes[i, j] = int(np.sum(arr))
                totals[i, j] = len(arr)

        with pm.Model() as model:
            if self._use_noncentered:
                mu_logit = pm.Normal("mu_logit", mu=0.0, sigma=2.0)
                tau = pm.HalfCauchy("tau", beta=self.kappa_prior_beta)

                offset = pm.Normal(
                    "offset", mu=0.0, sigma=1.0, shape=(n_seg, n_var)
                )

                logit_theta = pm.Deterministic(
                    "logit_theta", mu_logit + tau * offset
                )

                theta = pm.Deterministic(
                    "theta", pm.math.sigmoid(logit_theta)
                )

            else:
                mu_global = pm.Beta(
                    "mu_global",
                    alpha=self.prior_alpha,
                    beta=self.prior_beta,
                )
                kappa = pm.HalfCauchy("kappa", beta=self.kappa_prior_beta)

                alpha_seg = pm.Deterministic("alpha_seg", mu_global * kappa)
                beta_seg = pm.Deterministic(
                    "beta_seg", (1.0 - mu_global) * kappa
                )

                theta = pm.Beta(
                    "theta",
                    alpha=alpha_seg,
                    beta=beta_seg,
                    shape=(n_seg, n_var),
                )

            pm.Binomial(
                "obs",
                n=totals,
                p=theta,
                observed=successes,
            )

            trace = pm.sample(
                draws=n_draws,
                tune=1000,
                chains=4,
                target_accept=0.9,
                progressbar=False,
                return_inferencedata=True,
            )

        return trace

    def _extract_by_segment(
        self, trace: az.InferenceData, n_draws: int
    ) -> np.ndarray:
        """Extract per-segment posterior samples from trace."""

        theta_vals = trace.posterior["theta"].values
        total_draws, n_seg, n_var = (
            theta_vals.shape[0] * theta_vals.shape[1],
            theta_vals.shape[2],
            theta_vals.shape[3],
        )
        flat = theta_vals.reshape(total_draws, n_seg, n_var)

        if total_draws >= n_draws:
            idx = np.random.choice(total_draws, size=n_draws, replace=False)
            return flat[idx]
        else:
            warnings.warn(
                f"Requested {n_draws} draws but only {total_draws} available "
                f"after sampling. Returning all available draws.",
                UserWarning,
                stacklevel=3,
            )
            return flat

    def _run_health_checks(self, trace: az.InferenceData) -> None:
        """
        Run mandatory post-sampling diagnostics.

        Divergences > 100  -> RuntimeError (posterior not trustworthy)
        Divergences > 0    -> warning in _health_warnings
        R-hat > 1.01       -> warning
        ESS bulk < 400     -> warning

        Also checks for tau/kappa collapse.
        """

        divergences = int(trace.sample_stats["diverging"].values.sum())

        if divergences > 100:
            raise RuntimeError(
                f"MCMC produced {divergences} divergences. "
                f"Posterior is not trustworthy. "
                f"Try increasing n_draws, checking for data pathologies, "
                f"or switching parameterization."
            )
        elif divergences > 0:
            msg = (
                f"{divergences} MCMC divergences detected. "
                f"Results may be slightly unreliable. "
                f"Consider increasing n_draws."
            )
            self._health_warnings.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

        try:
            summary = az.summary(trace, var_names=["theta"], kind="stats")
            bad_rhat = summary[summary["r_hat"] > 1.01]
            if len(bad_rhat) > 0:
                msg = (
                    f"R-hat > 1.01 for {len(bad_rhat)} parameters — "
                    f"chains have not fully converged. Increase n_draws."
                )
                self._health_warnings.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=3)

            low_ess = summary[summary["ess_bulk"] < 400]
            if len(low_ess) > 0:
                msg = (
                    f"Low effective sample size (ESS < 400) for "
                    f"{len(low_ess)} parameters. Increase n_draws."
                )
                self._health_warnings.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=3)
        except Exception:
            pass

        self._check_concentration_collapse(trace)

        self._check_posterior_widths(trace)

    def _check_concentration_collapse(self, trace: az.InferenceData) -> None:
        """Warn if the concentration parameter collapses near zero."""
        param_name = "tau" if self._use_noncentered else "kappa"
        try:
            conc_vals = trace.posterior[param_name].values.flatten()
            conc_mean = float(conc_vals.mean())
            if conc_mean < 0.05:
                msg = (
                    f"Concentration parameter {param_name} collapsed near zero "
                    f"(posterior mean={conc_mean:.4f}). "
                    f"Segments are being treated as nearly identical. "
                    f"This is statistically correct if segments genuinely do not differ. "
                    f"If you believe segments differ, collect more data per segment."
                )
                self._health_warnings.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=4)
        except Exception:
            pass

    def _check_posterior_widths(self, trace: az.InferenceData) -> None:
        """Warn on segments with very wide credible intervals."""
        WIDTH_THRESHOLD = 0.5
        try:
            theta_vals = trace.posterior["theta"].values
            # Shape: (chain, draw, n_seg, n_var)
            for i, seg in enumerate(self._segment_names):
                seg_samples = theta_vals[:, :, i, :].flatten()
                width = (
                    np.percentile(seg_samples, 97.5)
                    - np.percentile(seg_samples, 2.5)
                )
                if width > WIDTH_THRESHOLD:
                    msg = (
                        f"Segment '{seg}': very wide 95% credible interval "
                        f"({width:.3f} on probability scale). "
                        f"Estimate is prior-dominated — insufficient data."
                    )
                    self._segment_warnings[seg].append(msg)
        except Exception:
            pass

    @property
    def segment_names(self) -> list[str]:
        """Sorted segment names. Available after fit()."""
        return list(self._segment_names)

    @property
    def all_warnings(self) -> dict:
        """All warnings from pre-flight and health checks."""
        return {
            "health": list(self._health_warnings),
            "segments": {
                seg: list(msgs)
                for seg, msgs in self._segment_warnings.items()
            },
        }