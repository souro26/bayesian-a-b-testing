# Bayesian A/B Testing — v2.0 (Sequential Decision Engine)

A Bayesian A/B testing system that treats experimentation as a **sequential decision-making problem**, rather than a one-time statistical inference task.

## v1.0 Baseline

v1.0 implemented a Bayesian A/B testing baseline using a Beta–Binomial model.
It focused on posterior inference for conversion rates and provided quantities such as
the probability that variant B outperforms A, expected lift, and credible intervals.

This established a solid inferential foundation, but it remained a **static analysis**:
it did not address how experiments should be evaluated sequentially or when a decision
should be made in practice.

## The Problem

Most A/B testing implementations focus on **static inference**:
they estimate conversion rates after collecting a fixed amount of data.

This approach has two major issues in practice:
- Experiments often run **longer than necessary**, wasting traffic and time.
- Statistical significance does not guarantee **practical or business relevance**.

v1.0 of this project addressed Bayesian inference, but it did not answer the key operational question:

**“When should we stop the experiment, and what should we do next?”**

## What Changed in v2.0

v2.0 upgrades the system from a static Bayesian analysis to a **sequential decision engine**.

Key changes:
- The experiment is evaluated **day by day**, not only at a fixed horizon.
- Decisions are made explicitly using Bayesian decision principles.
- The system can **stop early** when further data collection is no longer useful.

Instead of reporting only probabilities or intervals, v2.0 outputs one of three actions:
- `SHIP_B` — roll out the treatment
- `STOP` — stop the experiment and keep the control
- `CONTINUE` — collect more data

## Decision Semantics

- Variant A (control) is assumed to be the default.
- The system only ships B if evidence is strong and risk is acceptable.
- Otherwise, the experiment is stopped and A is kept by default.

This reflects common industry experimentation practice.

## How v2.0 Makes Decisions

The decision engine combines three ideas:

### 1. Bayesian Updating
Conversion rates are modeled using a Beta–Binomial model, allowing exact and fast posterior updates as new data arrives.

### 2. Practical Significance (ROPE)
Small effects may be statistically real but operationally irrelevant.
A Region of Practical Equivalence (ROPE) is used to distinguish meaningful improvements from noise.

### 3. Risk Awareness (Expected Loss)
Before shipping a variant, the system estimates the **expected loss** if the decision were wrong.
This prevents premature rollouts when uncertainty is still costly.

A decision is made only when confidence is high **and** downside risk is acceptable.

## Demo

The notebook `v2.0_demo.ipynb` demonstrates the system in action.

It simulates an A/B experiment where data arrives sequentially and shows:
- how evidence accumulates over time
- when the system decides to stop
- which action is taken

To run the demo:
1. Open `v2.0_demo.ipynb` in Google Colab
2. Run all cells from top to bottom
3. Observe the printed decisions and the decision-over-time plot

## Future Work (v3.0)

Potential extensions include:
- Threshold tuning via large-scale simulation
- Hierarchical models for segment-level experiments
- Support for non-binary metrics using PyMC
- Explicit rollback and multi-variant decision logic

These are intentionally left out of v2.0 to keep the decision behavior transparent and well-understood.
