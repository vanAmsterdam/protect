"""
test the demo as an integration test
"""

import pytest

import numpy as np
import numpyro

import pandas as pd
from jax import random

from causalprotect.models import PROTECTModel, tutorial_model
from causalprotect.inference import PROTECTInference
from causalprotect.utils import load_yaml
from causalprotect.utils import summary_likelihoods_to_df

NUM_FOLDS = 2
RNG_SEED = 1234
NUM_CHAINS = 4

numpyro.set_host_device_count(16)


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


def test_forward(model, data):
    # run the model forward once with a random seed
    output = model.sample(123, data)
    assert output.shape == (200,)


@pytest.fixture(scope="module")
def protector(data, model):
    protector = PROTECTInference(model, data)
    return protector


def test_inference(protector, rng_key):
    # run inference on original data
    rng_key, k_mcmc = random.split(rng_key)
    protector.run_inference(
        k_mcmc,
        mcmc_kwargs={
            "num_samples": 50,
            "num_warmup": 10,
            "num_chains": NUM_CHAINS,
            "chain_method": "parallel",
            "progress_bar": False,
        },
    )


## model checks
# TODO: implement baselines, needs to use better likelihood calculation
# def test_baselines(protector, rng_key):
#     # setup baselines
#     rng_key, k_baselines = random.split(rng_key)
#     protector.infer_baselines(rng_key)
#     all_lls = protector.summary_likelihoods()
#     lldf = summary_likelihoods_to_df(all_lls)
#     print(lldf.loc['baseline', 'sum'])


# run cross validated model checks
def test_model_checks(protector, rng_key, data):
    rng_key, k_model_checks = random.split(rng_key)
    model_check_result, _ = protector.model_checks(
        k_model_checks,
        num_folds=2,
        verbose=True,
        inference_mcmc_kwargs={
            "progress_bar": False,
            "num_samples": 50,
            "num_warmup": 50,
            "num_chains": 1,
        },
        num_global_samples=10,
        grid_kwargs={'K': 10},
        baseline_mcmc_kwargs={
            "progress_bar": False,
            "num_samples": 50,
            "num_warmup": 50,
            "num_chains": 1,
        },
    )

    # check the log likelihoods
    lldf = summary_likelihoods_to_df(model_check_result, dimnames=["fold"])

    num_obs_sites = len(protector.protect_model.obs_vars) + 4 # add 3 for txy_joint, joint, tx_enum and y_enum
    # 5 pp modes always: joint, no_tx, no_txy, no_y (excluding baseline as this is counted separately)
    # 3 for every proxyj: no_proxyj, no_proxyj_no_tx, no_proxyj_no_y
    num_ppmodes = 4 + 3 * len(protector.protect_model.f_proxies)

    # baseline ppmode has num_obs_sites - 2 obs_sites (not tx_enum and y_enum)
    num_baseline_obs_sites = num_obs_sites - 2

    expected_nrow_per_fold_split = num_baseline_obs_sites # 
    expected_nrow_per_fold_split += num_obs_sites * num_ppmodes
    expected_nrow = expected_nrow_per_fold_split * NUM_FOLDS * 2 # 2 splits: train / test

    assert lldf.shape == (expected_nrow, 10)

    # check first line
    num_obs = data["age"].shape[0]
    llrow1 = lldf.iloc[0,]
    assert llrow1["n_obs"] == int(num_obs / NUM_FOLDS)
    assert llrow1["sum"] - (num_obs / NUM_FOLDS) * llrow1["mean"] == pytest.approx(0., abs=1e-3)

    # check if all columns of the data frame have no missing values and whether all values are finite
    assert lldf.isnull().sum().sum() == 0
    num_cols = ['max', 'mean', 'min', 'n_obs', 'std', 'sum']
    assert np.isfinite(lldf[num_cols].values).all()

    # check if vmap is working
    model_check_result_vmap, _ = protector.model_checks(
        k_model_checks,
        num_folds=2,
        verbose=True,
        inference_mcmc_kwargs={
            "progress_bar": False,
            "num_samples": 50,
            "num_warmup": 50,
            "num_chains": 1,
        },
        num_global_samples=10,
        grid_kwargs={'K': 10, 'accelerator': 'vmap'},
        baseline_mcmc_kwargs={
            "progress_bar": False,
            "num_samples": 50,
            "num_warmup": 50,
            "num_chains": 1,
        },
    )

# TODO: add marginalization here
# k_postpred, k = random.split(k)
# k_postpred = random.PRNGKey(1)
# # converged, hrs = protector.get_marginalized_hazard_ratio(k_postpred)
# converged, hrs = protector.get_marginalized_hazard_ratio(k_postpred, num_local_draws=10)
# # assert all(converged)
# if not all(converged):
#     print(f"WARNING, not all samples converged, {np.sum(converged)} out of {len(converged)}")
# print(f"marginalized HR: {hrs.mean():.3f}, 95% HDI: {az.hdi(np.array(hrs), 0.95)}")
