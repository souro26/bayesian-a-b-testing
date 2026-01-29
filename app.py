import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.sequential_decision import sequential_ab_decision

st.set_page_config(page_title="Sequential Bayesian A/B Test", layout="centered")

st.title("Sequential Bayesian A/B Testing (v2.0)")
st.caption("A lightweight demo of a Bayesian decision engine with early stopping.")

st.sidebar.header("Simulation Parameters")

# --- User inputs ---
prior_alpha = st.sidebar.number_input("Prior α", value=1.0, min_value=0.1)
prior_beta = st.sidebar.number_input("Prior β", value=1.0, min_value=0.1)

true_p_a = st.sidebar.number_input("True p(A)", value=0.05, format="%.4f")
true_p_b = st.sidebar.number_input("True p(B)", value=0.06, format="%.4f")

users_per_day = st.sidebar.number_input("Users per day", value=1000, step=100)
max_days = st.sidebar.number_input("Max days", value=30, step=1)

run = st.sidebar.button("Run Simulation")

# --- Run simulation ---
if run:
    np.random.seed(42)

    successes_a = successes_b = 0
    trials_a = trials_b = 0

    days = []
    expected_lifts = []
    decisions = []

    st.subheader("Decision Log")

    for day in range(1, max_days + 1):
        daily_a = np.random.binomial(users_per_day, true_p_a)
        daily_b = np.random.binomial(users_per_day, true_p_b)

        successes_a += daily_a
        trials_a += users_per_day

        successes_b += daily_b
        trials_b += users_per_day

        result = sequential_ab_decision(
            successes_a=successes_a,
            trials_a=trials_a,
            successes_b=successes_b,
            trials_b=trials_b,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )

        days.append(day)
        expected_lifts.append(result["expected_lift"])
        decisions.append(result["decision"])

        st.write(
            f"Day {day:02d} → Lift: {result['expected_lift']:.4f} | "
            f"Decision: {result['decision']}"
        )

        if result["decision"] != "CONTINUE":
            st.success(f"Experiment concluded: {result['decision']}")
            break

    # --- Plot ---
    st.subheader("Evidence Over Time")

    fig, ax = plt.subplots()
    ax.plot(days, expected_lifts, marker="o")
    ax.axhline(0)
    ax.axvline(days[-1], linestyle="--")

    ax.set_xlabel("Day")
    ax.set_ylabel("Expected Lift (B − A)")
    ax.set_title("Sequential Decision Behavior")

    st.pyplot(fig)

