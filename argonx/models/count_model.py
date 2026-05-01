import numpy as np
import pymc as pm
import arviz as az
import warnings
from .base_model import BaseModel


class PoissonModel(BaseModel):
    """
    Bayesian model for count data using Poisson likelihood.

    Uses a Gamma prior on the Poisson rate parameter (lambda).
    """

    def __init__(self, lam_prior_alpha=1.0, lam_prior_beta=1.0):
        super().__init__()
        self.lam_prior_alpha = lam_prior_alpha
        self.lam_prior_beta = lam_prior_beta

    def _validate_input(self, data) -> None:
        super()._validate_input(data)
        for k, v in data.items():
            if (v < 0).any():
                raise ValueError(f"{k} must contain non-negative values")
            if not np.issubdtype(v.dtype, np.integer):
                raise ValueError(f"{k} must contain integer values")

    def sample_posterior(self, n_draws: int = 2000) -> np.ndarray:
        """
        Return posterior samples for the Poisson rate parameter.

        Parameters
        ----------
        n_draws : int, optional
            Number of posterior draws per variant, by default 2000.

        Returns
        -------
        np.ndarray
            Array of posterior samples of shape (n_draws, n_variants).
        """
        if self.data is None or self.variant_names is None:
            raise ValueError("Call fit() before sampling")

        if n_draws == 0:
            return np.empty((0, len(self.variant_names)))

        samples = []

        for variant in self.variant_names:
            data = self.data[variant]

            with pm.Model():
                lam = pm.Gamma(
                    "lam",
                    alpha=self.lam_prior_alpha,
                    beta=self.lam_prior_beta,
                )
                pm.Poisson("obs", mu=lam, observed=data)

                trace = pm.sample(
                    draws=n_draws,
                    tune=min(1000, n_draws),
                    chains=1,
                    progressbar=False,
                    random_seed=42,
                )

            lam_samples = trace.posterior["lam"].values.flatten()
            samples.append(lam_samples)

        return np.column_stack(samples)


class HierarchicalPoissonModel(BaseModel):
    """
    Hierarchical Bayesian model for count data across segments.

    Uses a Poisson likelihood with partial pooling on the log-rate across
    segments. Modelling in log space is natural for Poisson rates — the
    log link ensures rates are always positive without constraints, and
    the Normal prior on the log rate is the standard choice for
    log-linear hierarchical models.

    Parameters
    ----------
    log_lam_prior_mean : float, optional
        Mean of the normal prior on the global log-rate, by default 0.0.
    log_lam_prior_sd : float, optional
        Standard deviation of the normal prior on the global log-rate, by default 5.0.
    tau_prior_beta : float, optional
        Beta parameter for the HalfCauchy prior on the across-segment standard deviation, by default 1.0.
    priors : dict | None, optional
        Dictionary to override default priors, by default None.
    """

    MIN_SEGMENT_SIZE = 10

    def __init__(
        self,
        log_lam_prior_mean: float = 0.0,
        log_lam_prior_sd: float = 5.0,
        tau_prior_beta: float = 1.0,
        priors: dict | None = None,
    ):
        super().__init__()
        self.log_lam_prior_mean = log_lam_prior_mean
        self.log_lam_prior_sd = log_lam_prior_sd
        self.tau_prior_beta = tau_prior_beta

        if priors:
            _allowed = {
                'log_lam_prior_mean', 'log_lam_prior_sd', 'tau_prior_beta'
            }
            unknown = set(priors) - _allowed
            if unknown:
                raise ValueError(
                    f"Unknown prior keys: {unknown}. Allowed: {_allowed}"
                )
            self.log_lam_prior_mean = priors.get(
                'log_lam_prior_mean', self.log_lam_prior_mean
            )
            self.log_lam_prior_sd = priors.get(
                'log_lam_prior_sd', self.log_lam_prior_sd
            )
            self.tau_prior_beta = priors.get('tau_prior_beta', self.tau_prior_beta)

        self._segment_names: list[str] = []
        self._variant_names_hier: list[str] = []
        self._segment_data: dict[str, dict[str, np.ndarray]] = {}
        self._use_noncentered: bool = True
        self._segment_warnings: dict[str, list[str]] = {}
        self._health_warnings: list[str] = []
        self._trace = None

    def fit(self, data: dict[str, dict[str, np.ndarray]]) -> None:
        """
        Fit hierarchical model to segmented count data.

        Parameters
        ----------
        data : dict[str, dict[str, np.ndarray]]
            Nested dictionary mapping segments and variants to count data arrays.
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
        if not isinstance(data, dict):
            raise TypeError(
                "Hierarchical model expects dict[str, dict[str, np.ndarray]]. "
                f"Got {type(data).__name__}."
            )
        if len(data) < 2:
            raise ValueError("At least two segments required.")

        variant_sets = []
        for seg, variants in data.items():
            if not isinstance(variants, dict):
                raise TypeError(
                    f"Segment '{seg}' must map to dict[str, np.ndarray]."
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
                        f"Segment '{seg}', variant '{var}': expected 1D array."
                    )
                if len(arr) == 0:
                    raise ValueError(
                        f"Segment '{seg}', variant '{var}' has no observations."
                    )
                if np.isnan(arr).any():
                    raise ValueError(
                        f"Segment '{seg}', variant '{var}' contains NaN values."
                    )
                if (arr < 0).any():
                    raise ValueError(
                        f"Segment '{seg}', variant '{var}' must be non-negative."
                    )
                if not np.issubdtype(arr.dtype, np.integer):
                    raise ValueError(
                        f"Segment '{seg}', variant '{var}' must be integer counts."
                    )

        if len(set(variant_sets)) > 1:
            raise ValueError("All segments must have identical variant sets.")

    def _preflight_checks(self) -> None:
        for seg in self._segment_names:
            min_obs = min(
                len(self._segment_data[seg][var])
                for var in self._variant_names_hier
            )
            if min_obs < self.MIN_SEGMENT_SIZE:
                msg = (
                    f"Segment '{seg}' has only {min_obs} observations in its "
                    f"smallest variant cell. Posterior will be prior-dominated. "
                    f"Partial pooling will borrow heavily from other segments. "
                    f"Treat estimates with caution."
                )
                self._segment_warnings[seg].append(msg)
                warnings.warn(
                    f"Segment '{seg}': only {min_obs} observations. "
                    f"Posterior will be prior-dominated.",
                    UserWarning,
                    stacklevel=3,
                )

    def sample_posterior(self, n_draws: int = 1000) -> np.ndarray:
        """
        Return population-level posterior samples.

        Parameters
        ----------
        n_draws : int, optional
            Number of draws to return, by default 1000.

        Returns
        -------
        np.ndarray
            Array of population-level posterior samples.
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
        Return per-segment posterior samples.

        Parameters
        ----------
        n_draws : int, optional
            Number of draws to return, by default 1000.

        Returns
        -------
        np.ndarray
            Array of posterior samples of shape (n_draws, n_segments, n_variants).
        """
        if self._trace is None:
            self._trace = self._run_mcmc(n_draws=n_draws)
            self._run_health_checks(self._trace)

        return self._extract_by_segment(self._trace, n_draws)

    def _run_mcmc(self, n_draws: int) -> az.InferenceData:
        n_seg = len(self._segment_names)
        n_var = len(self._variant_names_hier)

        obs_arrays = [
            [self._segment_data[seg][var] for var in self._variant_names_hier]
            for seg in self._segment_names
        ]

        with pm.Model():
            log_lam_global = pm.Normal(
                "log_lam_global",
                mu=self.log_lam_prior_mean,
                sigma=self.log_lam_prior_sd,
            )
            tau = pm.HalfCauchy("tau", beta=self.tau_prior_beta)

            if self._use_noncentered:
                log_lam_offset = pm.Normal(
                    "log_lam_offset",
                    mu=0.0,
                    sigma=1.0,
                    shape=(n_seg, n_var),
                )
                log_lam_seg = pm.Deterministic(
                    "log_lam_seg", log_lam_global + tau * log_lam_offset
                )
            else:
                log_lam_seg = pm.Normal(
                    "log_lam_seg",
                    mu=log_lam_global,
                    sigma=tau,
                    shape=(n_seg, n_var),
                )

            # Rate is always positive — exp of log-rate
            lam_seg = pm.Deterministic("lam_seg", pm.math.exp(log_lam_seg))

            for i in range(n_seg):
                for j in range(n_var):
                    pm.Poisson(
                        f"obs_{i}_{j}",
                        mu=lam_seg[i, j],
                        observed=obs_arrays[i][j],
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
        """
        Extract lam_seg posterior. Returns (n_draws, n_segments, n_variants).
        """
        lam_vals = trace.posterior["lam_seg"].values
        total_draws = lam_vals.shape[0] * lam_vals.shape[1]
        flat = lam_vals.reshape(total_draws, lam_vals.shape[2], lam_vals.shape[3])

        if total_draws >= n_draws:
            idx = np.random.choice(total_draws, size=n_draws, replace=False)
            return flat[idx]
        else:
            warnings.warn(
                f"Requested {n_draws} draws but only {total_draws} available. "
                f"Returning all available draws.",
                UserWarning,
                stacklevel=3,
            )
            return flat

    def _run_health_checks(self, trace: az.InferenceData) -> None:
        divergences = int(trace.sample_stats["diverging"].values.sum())

        if divergences > 100:
            raise RuntimeError(
                f"MCMC produced {divergences} divergences. "
                f"Posterior is not trustworthy. "
                f"Try increasing n_draws or checking data for pathologies."
            )
        elif divergences > 0:
            msg = (
                f"{divergences} MCMC divergences. Results may be unreliable. "
                f"Consider increasing n_draws."
            )
            self._health_warnings.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

        try:
            summary = az.summary(trace, var_names=["lam_seg"], kind="stats")
            bad_rhat = summary[summary["r_hat"] > 1.01]
            if len(bad_rhat) > 0:
                msg = (
                    f"R-hat > 1.01 for {len(bad_rhat)} parameters. "
                    f"Chains have not converged. Increase n_draws."
                )
                self._health_warnings.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=3)

            low_ess = summary[summary["ess_bulk"] < 400]
            if len(low_ess) > 0:
                msg = (
                    f"Low ESS for {len(low_ess)} parameters. "
                    f"Increase n_draws."
                )
                self._health_warnings.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=3)
        except Exception:
            pass

        self._check_tau_collapse(trace)
        self._check_posterior_widths(trace)

    def _check_tau_collapse(self, trace: az.InferenceData) -> None:
        try:
            tau_mean = float(trace.posterior["tau"].values.mean())
            if tau_mean < 0.05:
                msg = (
                    f"tau collapsed near zero (posterior mean={tau_mean:.4f}). "
                    f"Segments are being treated as nearly identical. "
                    f"This is correct if segments genuinely do not differ. "
                    f"If you believe segments differ, collect more data per segment."
                )
                self._health_warnings.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=4)
        except Exception:
            pass

    def _check_posterior_widths(self, trace: az.InferenceData) -> None:
        WIDTH_THRESHOLD = 3.0  # wide in log-rate space
        try:
            lam_vals = trace.posterior["lam_seg"].values
            for i, seg in enumerate(self._segment_names):
                seg_samples = lam_vals[:, :, i, :].flatten()
                width = (
                    np.percentile(seg_samples, 97.5)
                    - np.percentile(seg_samples, 2.5)
                )
                if width > WIDTH_THRESHOLD:
                    msg = (
                        f"Segment '{seg}': wide 95% credible interval "
                        f"({width:.3f} in rate space). "
                        f"Estimate is prior-dominated — insufficient data."
                    )
                    self._segment_warnings[seg].append(msg)
        except Exception:
            pass

    @property
    def segment_names(self) -> list[str]:
        """
        Get sorted segment names.

        Returns
        -------
        list[str]
            List of segment names.
        """
        return list(self._segment_names)

    @property
    def all_warnings(self) -> dict:
        """
        Get all health and segment warnings.

        Returns
        -------
        dict
            Dictionary containing 'health' and 'segments' warning lists.
        """
        return {
            "health": list(self._health_warnings),
            "segments": {
                seg: list(msgs)
                for seg, msgs in self._segment_warnings.items()
            },
        }