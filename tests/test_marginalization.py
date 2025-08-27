# test marginalization of hazard ratio in power generalized weibull model
# compare with tests/rcode/marginal_hr.R

from causalprotect.inference import _marginalize_hazard_ratio_pgw
import pytest


import jax
from jax import numpy as jnp
from jax import random

RNG_SEED = 123
N = int(1e5)
B_TX_Y = 1.0

@pytest.fixture(scope="module")
def sample():
    rng_key = random.PRNGKey(RNG_SEED)

    lp_notx = random.normal(rng_key, (N,))
    lp_dotx = lp_notx + B_TX_Y

    sample = {
        "beta0": -2.3,
        "alpha0": -0.01,
        "nu0": 0.7,
        "lp_notx": lp_notx,
        "lp_dotx": lp_dotx,
        "b_tx_y": B_TX_Y
    }
    return sample

def test_marginalize_hazard_ratio_pgw(sample):
    rng_key = random.PRNGKey(RNG_SEED)
    
    # Test with a single sample
    result, t0, t1 = _marginalize_hazard_ratio_pgw(rng_key, sample, maxtime=100)
    assert result.converged
    assert result.position.shape == (4,)
    assert t0.shape == (N,)
    assert t1.shape == (N,)
    assert result.position[1] == pytest.approx(0.642, abs=2e-2)

