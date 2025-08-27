"""
testing for model checks
"""

import numpy as np
import jax
import numpyro
numpyro.set_platform("cpu")
numpyro.set_host_device_count(16)
import pytest

import pandas as pd
from jax import numpy as jnp, random

from causalprotect.inference import PROTECTInference
from causalprotect.models import PROTECTModel
from causalprotect.utils import load_yaml, summary_likelihoods_to_df

rng_key = random.PRNGKey(0)
NUM_OBS = 30
NUM_FOLDS = 2
NUM_GLOBAL_SAMPLES = 10
BASELINE_MCMC_KWARGS = {"num_warmup": 50, "num_samples": 50, "num_chains": 2, "chain_method": "sequential"}
INFERENCE_MCMC_KWARGS = {"num_warmup": 50, "num_samples": 50, "num_chains": 2, "chain_method": "sequential"}
POSTPRED_MCMC_KWARGS = {"num_warmup": 50, "num_samples": 50}

@pytest.fixture(scope="module")
def load_data():
    # read the simulated data
    data_pd = pd.read_csv("tests/testdata.csv")
    data_pd_head = data_pd.head(NUM_OBS)
    data = {k: np.array(v) for k, v in data_pd_head.items()}
    return data


@pytest.fixture(scope="module")
def load_metadata():
    return load_yaml("tests/metadata.yaml")


@pytest.fixture(scope="module")
def load_protect_model(load_data, load_metadata):
    from model import tutorial_model

    return PROTECTModel(
        tutorial_model, load_metadata, prior_spec="tests/priors.csv", data=load_data
    )

@pytest.fixture(scope="module")
def protector(load_data, load_protect_model):
    return PROTECTInference(load_protect_model, load_data)

@pytest.fixture(scope="module")
def test_model_checks_baseline_only(protector):

    result, in_train_mat = protector.model_checks(
        rng_key, num_folds = NUM_FOLDS, 
        verbose=True,
        do_baseline=True,
        do_postpred=False,
        num_global_samples=10,
        baseline_mcmc_kwargs=BASELINE_MCMC_KWARGS,
    )

    exected_pp_modes = {
        "baseline",
    }

    assert set(result.keys()) == exected_pp_modes

    return result, in_train_mat


@pytest.fixture(scope="module")
def test_model_checks_no_baseline(protector):

    result, in_train_mat = protector.model_checks(
        rng_key, num_folds = NUM_FOLDS, 
        verbose=True,
        do_baseline=False,
        do_postpred=True,
        num_global_samples=NUM_GLOBAL_SAMPLES,
        inference_mcmc_kwargs=INFERENCE_MCMC_KWARGS,
        postpred_mcmc_kwargs=POSTPRED_MCMC_KWARGS,
    )

    exected_pp_modes = {
        "joint",
        "no_y",
        "no_tx",
        "no_txy",
        "no_proxy1",
        "no_proxy1_no_y",
        "no_proxy1_no_tx",
        "no_proxy2",
        "no_proxy2_no_y",
        "no_proxy2_no_tx",
    }


    assert set(result.keys()) == exected_pp_modes

    return result, in_train_mat


@pytest.fixture(scope="module")
def test_model_checks_all(protector):

    result, in_train_mat = protector.model_checks(
        rng_key, num_folds = NUM_FOLDS, 
        verbose=True,
        do_baseline=True,
        # do_postpred=True,
        num_global_samples=NUM_GLOBAL_SAMPLES,
        baseline_mcmc_kwargs=BASELINE_MCMC_KWARGS,
        inference_mcmc_kwargs=INFERENCE_MCMC_KWARGS,
        postpred_mcmc_kwargs=POSTPRED_MCMC_KWARGS,
    )

    exected_pp_modes = {
        "baseline",
        "joint",
        "no_y",
        "no_tx",
        "no_txy",
        "no_proxy1",
        "no_proxy1_no_y",
        "no_proxy1_no_tx",
        "no_proxy2",
        "no_proxy2_no_y",
        "no_proxy2_no_tx",
    }

    assert set(result.keys()) == exected_pp_modes

    return result, in_train_mat

def test_model_check_evaluation(
        load_metadata,
        protector,
        test_model_checks_all,
):
    model_check_result, _ = test_model_checks_all
    model_check_evaluation = protector.protect_model.model_check_evaluation(model_check_result)
    f_proxies = load_metadata['f_proxies']
    obs_vars = f_proxies + ['y', 'tx']
    set1 = model_check_evaluation['set1']
    for obs_var in obs_vars:
        assert obs_var in set1, f"{obs_var} not in set1"
        assert 'passed' in set1[obs_var]
    
    set2 = model_check_evaluation['set2']
    for proxy in f_proxies:
        assert proxy in set2, f"{proxy} not in set2"
        proxy_res = set2[proxy]
        assert 'no_proxy' in proxy_res, f"'no_{proxy}' not in {proxy_res}"
        assert 'no_proxy_no_tx'in proxy_res
        assert 'no_proxy_no_y'in proxy_res
        assert 'passed' in proxy_res

    assert 'all_passed' in model_check_evaluation, "'all_passed' not in model_check_evaluation"

    # make the dataframe
    from causalprotect.utils import model_check_evaluation_to_df
    df = model_check_evaluation_to_df(model_check_evaluation)
    # check that the dataframe has the expected columns
    assert 'passed' in df.columns, "'passed' not in df columns"
    # check corret number of rows
    assert len(df) == len(obs_vars) + len(f_proxies), f"Expected {len(obs_vars) + len(f_proxies)} rows, got {len(df)}"



def test_pp_seed_consistency(
    test_model_checks_baseline_only,
    test_model_checks_no_baseline,
    test_model_checks_all,
):

    baseline_only_ll, baseline_only_intrain = test_model_checks_baseline_only
    no_baseline_ll, no_baseline_intrain = test_model_checks_no_baseline
    all_ll, all_intrain = test_model_checks_all

    # check that the in_train matrices are all the same
    assert np.allclose(baseline_only_intrain, all_intrain)
    assert np.allclose(no_baseline_intrain, all_intrain)

    # check that the likelihoods are all the same
    # check that the postpred results are consistent when using a single seed
    assert all_ll['baseline'][('test', 'joint')]['max'][0] == pytest.approx(
        baseline_only_ll['baseline'][('test', 'joint')]['max'][0],
        abs = 0.0001
    )
    assert all_ll['no_y'][('test', 'joint')]['max'][0] == pytest.approx(
        no_baseline_ll['no_y'][('test', 'joint')]['max'][0],
        abs = 0.0001
    )


def test_model_checks_mcmc(protector):

    result, in_train_mat = protector.model_checks(
        rng_key, num_folds = NUM_FOLDS, 
        verbose=True,
        do_baseline=False,
        do_postpred=['no_y'],
        pp_inference='mcmc',
        num_global_samples=NUM_GLOBAL_SAMPLES,
        inference_mcmc_kwargs=INFERENCE_MCMC_KWARGS,
        postpred_mcmc_kwargs=POSTPRED_MCMC_KWARGS,
    )

    exected_pp_modes = {
        "no_y",
    }


    assert set(result.keys()) == exected_pp_modes

