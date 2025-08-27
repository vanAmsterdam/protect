"""
testing for baseline calculation
"""

import numpy as np
import numpyro

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

import pandas as pd
import pytest
from jax import random

from causalprotect.inference import PROTECTInference
from causalprotect.models import PROTECTModel
from causalprotect.utils import load_yaml, summarize_likelihoods


@pytest.fixture(scope="module")
def load_data():
    # read the simulated data
    data_pd = pd.read_csv("tests/testdata.csv")
    data = {k: np.array(v) for k, v in data_pd.items()}
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


def test_baselines(protector):
    """
    test against likelihoods from R
    """
    rng_key = random.PRNGKey(0)
    _, lls_per_patient = protector.infer_baselines(rng_key=rng_key)
    baseline_lls = summarize_likelihoods(lls_per_patient)

    assert baseline_lls[("train", "proxy1")]["sum"] == pytest.approx(-126.835, abs=5e-1)
    assert baseline_lls[("train", "proxy2")]["sum"] == pytest.approx(-77.277, abs=5e-1)
    assert baseline_lls[("train", "tx")]["sum"] == pytest.approx(-115.644, abs=5e-1)
    assert baseline_lls[("train", "y")]["sum"] == pytest.approx(-367.102, abs=5e-1)

    # as all these likelihoods are completely independent (no shared parameters and independent prior),
    # they should sum up to the joint (which is a per-sample joint)
    assert baseline_lls[("train", "joint")]["sum"] == pytest.approx(
        -126.835 - 77.277 - 115.644 - 367.102, abs=5e-1
    )
    # txy should sum up to tx + y
    assert baseline_lls[("train", "txy")]["sum"] == pytest.approx(
        -115.644 - 367.102, abs=5e-1
    )

