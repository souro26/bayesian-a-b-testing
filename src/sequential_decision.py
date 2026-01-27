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
    """
    Returns posterior samples, expected loss, and a decision:
    STOP, CONTINUE, or SHIP_B
    """

