# argonx

![CI](https://github.com/souro26/bayesian-a-b-testing/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/argonx.svg)
![Downloads](https://img.shields.io/pypi/dm/argonx.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

argonx is a Bayesian A/B testing engine that tells you what to do — not just whether an effect exists.

Most tools give you p-values.
argonx gives you decisions:

• Which variant to ship
• How much you lose if you're wrong
• Whether it's safe to stop early

Built for real-world experiments where tradeoffs matter.

> If you've ever asked “should we ship this?” and got a p-value instead of an answer — this is for you.

---

## Install

```bash
pip install argonx
```

```bash
# development install
git clone https://github.com/souro26/bayesian-a-b-testing.git
cd bayesian-a-b-testing
pip install -e .
```

---

## Quick Start

```python
from argonx import Experiment

experiment = Experiment(
    data=df,
    variant_col='variant',
    primary_metric='revenue',
    guardrails=['page_load_ms'],
    lower_is_better={'page_load_ms': True},
    model='lognormal',
    guardrail_models={'page_load_ms': 'gaussian'},
    control='control',
)

result = experiment.run()
result.summary()
result.plot()
```

Ratio metrics via callable:

```python
experiment = Experiment(
    data=df,
    variant_col='variant',
    primary_metric=lambda df: df['clicks'] / df['impressions'],
    model='lognormal',
    control='control',
)
```

Segment-aware hierarchical inference:

```python
experiment = Experiment(
    data=df,
    variant_col='variant',
    segment_col='device_type',
    primary_metric='revenue',
    model='lognormal',
    control='control',
)

result = experiment.run()
result.summary()
result.segment_summary()
```

---

## Why argonx?

Because raw posteriors don’t help you decide.

argonx turns Bayesian inference into clear, decision-ready output:

```python
result.summary()
```

```
Best Variant: variant_b
P(best): 0.971

Expected loss if wrong: 0.0009
CVaR: 0.0021

Guardrail violation detected → REVIEW REQUIRED
```

No interpretation layer needed. The tradeoffs are explicit.

---

## Who is this for?

* Startups running A/B tests without expensive tools
* Engineers who want interpretable decisions, not black-box stats
* Teams dealing with tradeoffs (conversion vs latency, revenue vs churn)

---

## What It Computes

argonx focuses on decision-making under uncertainty — not just statistical significance.

| Metric                  | What it answers                                                    |
| ----------------------- | ------------------------------------------------------------------ |
| **P(variant is best)**  | Posterior probability of being the true winner across all variants |
| **Expected loss**       | Average downside if you ship the wrong variant                     |
| **CVaR**                | Tail risk — worst-case loss when things go wrong                   |
| **ROPE**                | Whether the effect is practically meaningful                       |
| **HDI**                 | Posterior interval of the effect                                   |
| **Joint probability**   | Probability all business conditions hold simultaneously            |
| **Composite score**     | Multi-metric business impact computed from posterior draws         |
| **Guardrail conflict**  | Detects tradeoffs (e.g. conversion ↑ but latency ↑)                |
| **Sequential stopping** | Stops when risk is low enough — not when sample size is arbitrary  |

---

## What `result.summary()` Looks Like

```
============================================================
EXPERIMENT RESULTS
============================================================

PRIMARY METRIC
----------------------------------------
Best Variant: variant_b
Expected lift:    +4.3% (95% HDI: +1.0% to +7.0%)
P(best): 0.971

RISK
----------------------------------------
Expected loss if wrong:          0.0009
CVaR (95th percentile loss):     0.0021
Risk level:                      low

PRACTICAL SIGNIFICANCE (ROPE)
----------------------------------------
Effect is OUTSIDE ROPE -- practically meaningful.
P(practical effect): 0.941

GUARDRAILS
----------------------------------------
  page_load_ms    [FAIL]  P(degraded)=0.912

GUARDRAIL CONFLICT DETECTED
----------------------------------------
Primary metric improves, but guardrail degrades.
Automatic decision blocked.

============================================================
DECISION
----------------------------------------
State:          conflict
Recommendation: REVIEW REQUIRED
Confidence:     low
============================================================
```

The framework does not make the decision.
It makes the tradeoffs explicit so you can.

---

## Models

| Model       | Use case                       | Data type       |
| ----------- | ------------------------------ | --------------- |
| `binary`    | Conversion rate, click-through | 0/1             |
| `lognormal` | Revenue, order value           | Positive skewed |
| `gaussian`  | Latency, scores                | Continuous      |
| `studentt`  | Robust to outliers             | Heavy-tailed    |
| `poisson`   | Counts, events                 | Integer         |

Hierarchical modeling is enabled automatically via `segment_col`.
Thin segments borrow strength without collapsing differences.

---

## Sequential Stopping

```python
from argonx.sequential import StoppingChecker

checker = StoppingChecker(
    loss_threshold=0.01,
    prob_best_min=0.95,
    min_sample_size=1000,
)

status = checker.update(
    samples=result.samples,
    variant_names=['control', 'variant_b'],
    control='control',
    n_users_per_variant=n_counts,
)

print(status.safe_to_stop)
print(status.users_needed)

checker.plot_trajectory()
```

argonx stops when expected loss is low enough — not when arbitrary sample sizes are reached.

---

## Examples

Worked examples in [`examples/`](examples/):

| Notebook               | Scenario                |
| ---------------------- | ----------------------- |
| Checkout redesign      | Guardrail conflict      |
| SaaS pricing           | Early stopping          |
| Clinical trial         | Heavy-tailed modeling   |
| Gaming matchmaking     | Multi-variant selection |
| Mobile personalisation | Hierarchical conflict   |

---

## Running Tests

```bash
pytest tests/unit/
pytest tests/math/
pytest tests/
```

---

## Contributing

Open an issue before major changes.
Run tests before PR.

---

## License

MIT
