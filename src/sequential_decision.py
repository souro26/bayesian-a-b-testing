import numpy as np
from scipy.stats import beta


def sequential_ab_decision(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    rope: float = 0.01,
    loss_threshold: float = 0.01,
    n_samples: int = 50_000,
):
    # Compute posterior parameters
    alpha_a = prior_alpha + successes_a
    beta_a = prior_beta + (trials_a - successes_a)

    alpha_b = prior_alpha + successes_b
    beta_b = prior_beta + (trials_b - successes_b)

    # Sample from posteriors
    samples_a = beta.rvs(alpha_a, beta_a, size=n_samples)
    samples_b = beta.rvs(alpha_b, beta_b, size=n_samples)

    # Compute lift distribution
    lift = samples_b - samples_a

    # ROPE to check for validity in chance 
    prob_lift_positive = np.mean(lift > rope)
    prob_lift_negative = np.mean(lift < -rope)
    prob_practically_equal = np.mean(
        (lift >= -rope) & (lift <= rope)
    )
    
    # Expected loss
    loss_if_ship_b = np.maximum(0, -lift)
    expected_loss = np.mean(loss_if_ship_b)

    # Decision rule
    if prob_lift_positive > 0.95 and expected_loss < loss_threshold:
        decision = "SHIP B"
    elif prob_practically_equal > 0.90:
        decision = "STOP"
    else:
        decision = "CONTINUE WITH A"


    # Return blocks
    return {
        "expected_lift": np.mean(lift),
        "expected_loss": expected_loss,
        "prob_lift_positive": prob_lift_positive,
        "prob_lift_negative": prob_lift_negative,
        "prob_practically_equal": prob_practically_equal,
        "decision": decision,
    }


