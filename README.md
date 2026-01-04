# Bayesian A/B Testing Tool

A Bayesian approach to A/B testing using **beta-binomial conjugate priors** for exact inference on conversion rates.

Unlike traditional frequentist methods, this tool provides:
- **Probability that variant B beats A** (not just a p-value)
- **Expected lift** with credible intervals
- **Clear decision recommendations** based on evidence strength

## 🚀 Try It Now

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y2BOdLZHQyZqb68ywsGIEMiOsoDK3YwE#scrollTo=5po-W9MOdLcM)

Click the badge above to run the interactive tool in Google Colab.

## 📊 Features

- **Interactive inputs** - directly enter visitor and conversion counts
- **Beta-binomial conjugate prior model** - exact Bayesian inference without MCMC
- **Monte Carlo sampling** - probability calculations via simulation
- **Posterior visualizations** - see the uncertainty in conversion rates
- **Lift distribution** - understand the magnitude of improvements
- **Decision logic** - confidence-based recommendations

## 🔧 Methodology

### Model
- **Prior**: Beta(α, β) distribution (uniform or weakly informative)
- **Likelihood**: Binomial(n, p) for conversion data
- **Posterior**: Beta(α + x, β + n - x) via conjugacy

### Metrics Computed
- **P(B > A)**: Probability variant B has a higher conversion rate
- **Expected lift**: Mean relative improvement of B over A
- **95% Credible Interval**: Range of plausible lift values
- **Decision recommendation**: Based on evidence strength and practical significance

## 📈 Usage

1. Open the Colab notebook using the badge above
2. Adjust input parameters:
   - `n_A`, `x_A`: Visitors and conversions for variant A (control)
   - `n_B`, `x_B`: Visitors and conversions for variant B (treatment)
   - Prior selection: Uniform (uninformative) or weakly informative
3. Run all cells (Runtime → Run all)
4. View results summary and visualizations

## 🛠️ Requirements
```
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

## 📝 Example Output
```
==================================================
BAYESIAN A/B TEST RESULTS
==================================================

📈 Probability B Beats A:  94.3%
📊 Expected Relative Lift: +8.32%
🎯 95% Credible Interval:  [+5.1%, +11.6%]
💰 Absolute Difference:    +0.0219 (percentage points)

────────────────────────────────────────────────────────────
DECISION: 🚀 LAUNCH VARIANT B
Confidence: STRONG
Reasoning: High confidence that B is better, with positive minimum lift.
==================================================
```

## 🎓 What I Learned

Building this tool helped me understand:
- Bayesian inference and conjugate priors
- Beta-binomial models for binary outcomes
- Monte Carlo sampling methods
- Interpretation of credible intervals vs confidence intervals
- Decision-making under uncertainty

## 🗺️ Roadmap

Future improvements planned:

- [ ] **Sequential testing** - update posteriors as data arrives in real-time
- [ ] **Continuous metrics** - support for revenue, time-on-site (Normal-Normal model)
- [ ] **Multi-variant testing** - A/B/C/D/... testing
- [ ] **Prior sensitivity analysis** - visualize impact of different priors
- [ ] **PyMC integration** - handle non-conjugate models and hierarchical structures
- [ ] **Power analysis** - sample size calculations
- [ ] **Expected loss** - quantify cost of wrong decisions

## 📚 Resources

- [Bayesian A/B Testing by Evan Miller](https://www.evanmiller.org/bayesian-ab-testing.html)
- [PyMC Documentation](https://www.pymc.io/)
- [Conjugate Prior - Wikipedia](https://en.wikipedia.org/wiki/Conjugate_prior)
- [Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution)

## 📄 License

MIT License - feel free to use and modify for your own projects

## 🤝 Contributing

Feedback and contributions welcome! Feel free to:
- Open an issue for bugs or suggestions
- Submit a PR for improvements
- Reach out with questions about Bayesian methods

---

**Built while learning Bayesian statistics** | Preparing to contribute to [PyMC](https://github.com/pymc-devs/pymc) for GSoC 2026


