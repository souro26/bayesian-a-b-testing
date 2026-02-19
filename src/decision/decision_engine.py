import numpy as np

def decision_engine(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    rope: float = 0.01,
    loss_threshold: float = 0.01,
):
    """
    Generic Bayesian decision engine.

    Takes posterior samples for two variants and returns
    lift statistics and a decision.
    """

    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)

    if samples_a.shape != samples_b.shape:
        raise ValueError("samples_a and samples_b must have the same shape.")

    if samples_a.size == 0:
        raise ValueError("Samples must not be empty.")

    lift = samples_b - samples_a

    expected_lift = np.mean(lift)

    prob_lift_positive = np.mean(lift > rope)
    prob_lift_negative = np.mean(lift < -rope)
    prob_equal = np.mean((lift >= -rope) & (lift <= rope))

    loss_if_ship_b = np.maximum(0, -lift)
    expected_loss = np.mean(loss_if_ship_b)

    if prob_lift_positive > 0.95 and expected_loss < loss_threshold:
        decision = "SHIP_B"
    elif prob_equal > 0.90:
        decision = "STOP"
    else:
        decision = "CONTINUE"

    return {
        "expected_lift": expected_lift,
        "expected_loss": expected_loss,
        "prob_lift_positive": prob_lift_positive,
        "prob_lift_negative": prob_lift_negative,
        "prob_equal": prob_equal,
        "decision": decision,
    }
