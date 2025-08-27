"""
testing for main module: protector
"""

import arviz as az
import numpy as np
import jax
import numpyro
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
import pytest

import pandas as pd
from jax import numpy as jnp, random

k = random.PRNGKey(0)
from causalprotect.inference import PROTECTInference
from causalprotect.models import PROTECTModel
from causalprotect.utils import load_yaml, summary_likelihoods_to_df

NUM_OBS = 30
NUM_FOLDS = 2
NUM_CHAINS = 2
NUM_SAMPLES = 50
NUM_LOCAL_DRAWS = 20

def test_slice_posterior_for_pp():
    from causalprotect.inference import _slice_posterior_for_pp

    num_chains = 2
    num_samples = 20
    num_samples_out = 10

    # test with group_by_chain=False
    samples = {"beta0": jnp.arange(num_chains * num_samples)}
    global_sample_shape = (num_chains * num_samples,)
    sliced = _slice_posterior_for_pp(samples, global_sample_shape, num_samples_out=num_samples_out, group_by_chain=False)
    assert sliced["beta0"].shape == (num_samples_out,)

    # test with group_by_chain=True
    samples = {"beta0": jnp.arange(num_chains * num_samples).reshape((num_chains, num_samples))}
    global_sample_shape = (num_chains, num_samples)
    sliced = _slice_posterior_for_pp(samples, global_sample_shape, num_samples_out=num_samples_out, group_by_chain=True)
    assert sliced["beta0"].shape == (num_chains, num_samples_out // num_chains)

    # test that it throws an error if num_samples_out is bigger than num_chains * num_samples
    with pytest.raises(ValueError):
        _slice_posterior_for_pp(samples, global_sample_shape, num_samples_out=num_chains * num_samples + 1, group_by_chain=False)
    # test that it throws an error if num_samples_out is not divisible by num_chains
    with pytest.raises(ValueError):
        _slice_posterior_for_pp(samples, global_sample_shape, num_samples_out=num_chains * num_samples + 1, group_by_chain=True)



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


def test_data(load_data, load_protect_model):
    assert load_protect_model.check_data(load_data)



@pytest.fixture(scope="module")
def protector(load_data, load_protect_model):
    return PROTECTInference(load_protect_model, load_data)


class TestProtectInference:
    @pytest.fixture(scope="class")
    def inference_samples(self, protector):
        rng_key = random.PRNGKey(6)
        samples = protector.run_inference(
            rng_key,
            mcmc_kwargs={"num_warmup": 50, "num_samples": NUM_SAMPLES, "num_chains": NUM_CHAINS},
        )
        return samples

    def test_inference(self, inference_samples):
        expected_prm_names = [
            "Feps",
            "Fhat",
            "alpha0",
            "b_F_proxy1",
            "b_F_proxy2",
            "b_F_tx",
            "b_F_y",
            "b_Ftx_y",
            "b_age_F",
            "b_tx_y",
            "b_tx_y_marginal",
            "beta0",
            "eta_tx",
            "lp",
            "lp_dotx",
            "lp_notx",
            "mu_proxy1",
            "mu_proxy2",
            "mu_tx",
            "nu0",
            "F_mu",
            "F_sd",
        ]
        assert set(inference_samples.keys()) == set(expected_prm_names)
        assert inference_samples["b_tx_y"].shape == (NUM_CHAINS * NUM_SAMPLES, )

    def test_get_samples(self, protector, inference_samples):
        samples = protector.get_samples()
        assert samples is not None

    @pytest.fixture(scope="class")
    def postpred_noy(self, protector, inference_samples):
        postpred_samples = protector.run_postpred_mcmc(random.PRNGKey(7), inference_samples, pp_mode="no_y", num_local_draws=NUM_LOCAL_DRAWS)
        assert postpred_samples is not None
        assert set(postpred_samples.keys()) == set(inference_samples.keys())
        assert postpred_samples["b_tx_y"].shape == (NUM_LOCAL_DRAWS,)
        return postpred_samples


    def test_c_index(self, protector, postpred_noy):

        c_index = protector.get_c_index()

        assert c_index is not None
        assert c_index.shape == (NUM_LOCAL_DRAWS, )
        assert c_index.mean() > 0
        assert c_index.mean() < 1
