import numpy as np

def decision_engine(
    samples_a,
    samples_b,
    rope : float = 0.01,
    loss_threshold : float = 0.01,
):
    lift = samples_b - samples_a
    
    prob_lift_positive = np.mean(lift > rope)
    prob_lift_negative = np.mean(lift < -rope)
    prob_equal = np.mean((lift >= -rope) & (lift <= rope))

    loss_if_ship_b = np.maximum(0, -lift)
    expected_loss = np.mean(loss_if_ship_b)
    
    if prob_lift_positive > 0.95 and expected_loss < loss_threshold:
        decision = "Ship B"
    elif prob_equal> 0.90:
        decision = "Stop"
    else:
        decision = "Continue"
        
    return { 
           "expected lift" : np.mean(lift),
           "expected loss" : expected_loss,
           "prob lift positive" : prob_lift_positive,
           "prob lift negative" : prob_lift_negative,
           "prob equal" : prob_equal,
           "decision" : decision,
    }