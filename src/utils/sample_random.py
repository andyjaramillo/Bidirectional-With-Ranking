from scipy.stats import truncnorm
import numpy as np

# Define parameters for the UNTRUNCATED normal distribution
MEAN = 0.0
STD_DEV = 1.0
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0

EXPECTED_VALUE = 5

def samp_rand_norm():
    """
    Generate a random value using a Poisson distribution.
    
    Returns:
        int: Randomly sampled value.
    """
    return np.random.poisson(lam=EXPECTED_VALUE)
    
