"""
test the implementation of the power generalized weibull model in python versus R
the reference data for the pgw tests are created with the R code in tests/r_scripts/test_pgw.R
the implementation of the pgw model in R is in rcode/pgwsource.R, which is based on the official implementation
"""

import jax
import numpyro
import pandas as pd
import pytest
from jax import numpy as jnp
from jax import random
from numpyro import distributions as dist
from numpyro import sample
from numpyro.infer.hmc import NUTS
from numpyro.infer.mcmc import MCMC
from numpyro.infer.util import log_likelihood

from causalprotect.distributions import PowerGeneralizedWeibullLog as PGW
# test utils.optimize_pgw
from causalprotect.utils import optimize_pgw

# Global variable for the number of samples
NUM_SAMPLES = 400


@pytest.fixture(scope="module")
def load_data():
    df = pd.read_csv("tests/testdata.csv")
    time_cens = jnp.array(df["time_cens"].values.astype("float32"))
    tx = jnp.array(df["tx"].values.astype(bool))
    age = jnp.array(df["age"].values.astype("float32"))
    return time_cens, tx, age


def test_optimize_pgw(load_data):
    time_cens, tx, age = load_data
    Xmat = jnp.column_stack((tx, age))

    # run optimization
    result = optimize_pgw(time_cens, Xmat)

    # check convergence
    assert result.converged

    # check parameter values
    params = result.position
    assert params[0] == pytest.approx(-0.61243533, abs=1e-1)
    assert params[1] == pytest.approx(-0.09193421, abs=1e-1)
    assert params[2] == pytest.approx(0.15705516, abs=1e-1)
    assert params[3] == pytest.approx(0.83349349, abs=1e-1)
    assert params[4] == pytest.approx(-0.13842867, abs=1e-1)
    #                       mle         se          z
    # b_(Intercept) -0.61243533 0.18787749 -3.2597590
    # b_tx          -0.09193421 0.17646485 -0.5209775
    # b_age          0.15705516 0.07684773  2.0437190
    # a_(Intercept)  0.83349349 0.09926469  8.3966762
    # n_(Intercept) -0.13842867 0.05691890 -2.4320332

    # check likelihood
    objective_value_mean = result.objective_value
    assert objective_value_mean*time_cens.shape[0] == pytest.approx(364.7471, abs=0.1)
    #   code     maxgrad iter npar      like      aic      bic
    # 1    1 0.000165841   15    5 -364.7471 739.4943 755.9859


def marginal_pwg(time_cens):
    alpha0 = sample("alpha0", dist.Normal(0, 5))
    beta0 = sample("beta0", dist.Normal(0, 5))
    nu0 = sample("nu0", dist.Uniform(-5, 5))

    with numpyro.plate("obs", time_cens.shape[0]):
        sample("obs_y", PGW(0.0, beta0, alpha0, nu0), obs=time_cens)


@pytest.fixture(scope="module")
def run_mcmc(load_data):
    time_cens, *_ = load_data
    rng_mcmc = random.PRNGKey(0)
    mcmc = MCMC(NUTS(marginal_pwg), num_warmup=100, num_samples=NUM_SAMPLES)
    mcmc.run(rng_mcmc, time_cens)
    return mcmc


def test_log_likelihood(run_mcmc, load_data):
    time_cens, *_ = load_data
    samples = run_mcmc.get_samples(group_by_chain=False)
    lls = log_likelihood(marginal_pwg, samples, time_cens=time_cens, batch_ndims=1)
    lls_per_patient = {
        k: jax.scipy.special.logsumexp(v, axis=0) - jnp.log(NUM_SAMPLES)
        for k, v in lls.items()
    }
    lls_sums = {k: jnp.sum(v) for k, v in lls_per_patient.items()}
    assert lls_sums["obs_y"] == pytest.approx(-367.1022, abs=1.0)


def test_pgw_parameters(run_mcmc):
    samples = run_mcmc.get_samples(group_by_chain=False)
    assert samples is not None
    assert "alpha0" in samples
    assert "beta0" in samples
    assert "nu0" in samples

    # check the parameters
    assert samples["alpha0"].mean() == pytest.approx(0.8371022, abs=1e-1)
    assert samples["beta0"].mean() == pytest.approx(-0.6787658, abs=1e-1)
    assert samples["nu0"].mean() == pytest.approx(-0.1474001, abs=1e-1)



#                      mle         se         z
# b_(Intercept) -0.6787658 0.13993848 -4.850459
# a_(Intercept)  0.8371022 0.09883253  8.469906
# n_(Intercept) -0.1474001 0.05645491 -2.610935

# expected log-likelihood: -367.1022
