"""
test the post pred methods
"""

import pytest

import numpy as np
import numpyro

import pandas as pd
from jax import random

from protect.models import PROTECTModel, tutorial_model
from protect.inference import PROTECTInference
from protect.utils import load_yaml

RNG_SEED = 123
NUM_CHAINS = 2
NUM_LOCAL_DRAWS = 5

numpyro.set_host_device_count(NUM_CHAINS)

@pytest.fixture(scope="module")
def rng_key():
    rng_key = random.PRNGKey(RNG_SEED)
    return rng_key


# read the simulated data
@pytest.fixture(scope="module")
def data():
    data_pd = pd.read_csv("tests/testdata.csv")
    data = {k: np.array(v) for k, v in data_pd.items()}
    return data


@pytest.fixture(scope="module")
def model(data):
    metadata = load_yaml("tests/metadata.yaml")

    model = PROTECTModel(
        tutorial_model, metadata, prior_spec="tests/priors.csv", data=data
    )
    model.check_data(data)
    return model


@pytest.fixture(scope="module")
def protector(data, model):
    protector = PROTECTInference(model, data)
    return protector

def test_postpred(protector, rng_key):
    nobs = protector.default_control["N"]
    rng_key, sub_key = random.split(rng_key)
    # first run inference
    global_samples = protector.run_inference(
        rng_key=sub_key,
        mcmc_kwargs = dict(
            num_samples=50,
            num_warmup=10,
            num_chains=NUM_CHAINS,
            chain_method="parallel",
            progress_bar=False,
        )
    )
    
    # test that the shape of the samples is correct for no_tx ppmode
    rng_key, sub_key = random.split(rng_key)
    pp_samples_no_tx = protector.run_postpred_mcmc(
        sub_key,
        global_samples,
        pp_mode="no_tx",
        num_local_draws=NUM_LOCAL_DRAWS,
        group_by_chain=False,
        mcmc_kwargs={
            "num_samples": 50,
            "num_warmup": 10,
        },
    )

    # check shapes of samples
    assert pp_samples_no_tx['alpha0'].shape == (NUM_LOCAL_DRAWS, )
    assert pp_samples_no_tx['lp'].shape == (NUM_LOCAL_DRAWS, 2, nobs)
    
    ## test per-chain
    global_samples_by_chain = protector.mcmc.get_samples(group_by_chain=True)

    rng_key, sub_key = random.split(rng_key)
    pp_samples_by_chain = protector.run_postpred_mcmc(
        sub_key,
        global_samples_by_chain,
        num_local_draws=NUM_LOCAL_DRAWS,
        group_by_chain=True,
        mcmc_kwargs={
            "num_samples": 50,
            "num_warmup": 10,
            "num_chains": NUM_CHAINS,
            "chain_method": "parallel",
            "progress_bar": False,
        },
    )

    # check shapes of samples
    assert pp_samples_by_chain['alpha0'].shape == (NUM_CHAINS, NUM_LOCAL_DRAWS // NUM_CHAINS, )
    assert pp_samples_by_chain['Feps'].shape == (NUM_CHAINS, NUM_LOCAL_DRAWS // NUM_CHAINS, nobs)

    ## test not per-chain
    rng_key, sub_key = random.split(rng_key)
    pp_samples = protector.run_postpred_mcmc(
        sub_key,
        global_samples,
        num_local_draws=NUM_LOCAL_DRAWS,
        group_by_chain=False,
        mcmc_kwargs={
            "num_samples": 50,
            "num_warmup": 10,
        },
    )

    # check shapes of samples
    assert pp_samples['alpha0'].shape == (NUM_LOCAL_DRAWS, )
    assert pp_samples['Feps'].shape == (NUM_LOCAL_DRAWS, nobs)
