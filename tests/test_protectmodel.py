import pytest
import pandas as pd
import numpy as np

from causalprotect.models import PROTECTModel, tutorial_model
from causalprotect.utils import load_yaml
from jax.random import PRNGKey

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
def load_protect_model(load_metadata):
    metadata = load_metadata
    metadata['local_prms'] = ['Feps']

    return PROTECTModel(
        tutorial_model, metadata, prior_spec="tests/priors.csv"
    )


def test_model_sample(load_data, load_protect_model):
    data = load_data
    output = load_protect_model.sample(123, data=data)
    assert output.shape == data["age"].shape
    assert output.mean() == pytest.approx(-0.11569660172999999, abs=1e-5)


def test_simulate_data(load_protect_model):
    nsim = 100000
    simprms = {'b_age_F': -1.0,
    'b_F_proxy1': -2.0,
    'mu_proxy1': 0.0,
    'b_F_proxy2': -1.5,
    'mu_proxy2': -2.5,
    'b_F_tx': 1.5,
    'mu_tx': -0.22,
    'b_F_y': -1.0,
    'b_tx_y': -0.02,
    'b_Ftx_y': -0.12,
    'alpha0': 0.74,
    'beta0': -0.28,
    'nu0': -0.06,
    }

    k = PRNGKey(1)
    data = load_protect_model.simulate_from_prms(k, simprms, nsim, maxtime=10)
    assert data is not None
    assert data['age'].shape[0] == nsim
    # values tested against nsim=1e6, k=PRGNKey(0)
    assert data['age'].mean() == pytest.approx(0.0, abs=1e-2)
    assert data['proxy1'].mean() == pytest.approx(0.2805, abs=1e-2)
    assert data['proxy2'].mean() == pytest.approx(0.8455, abs=1e-2)
    assert data['tx'].mean() == pytest.approx(0.7192, abs=1e-2)
    assert data['Fhat'].mean() == pytest.approx(0.4999, abs=1e-2)
    assert data['eta_tx'].mean() == pytest.approx(0.9699, abs=1e-2)
    assert data['b_tx_y_marginal'].mean() == pytest.approx(-0.0789, abs=1e-2)
    assert data['lp'].mean() == pytest.approx(-0.5598, abs=1e-2)
    assert data['lp_notx'].mean() == pytest.approx(-0.4999, abs=1e-2)
    assert data['lp_dotx'].mean() == pytest.approx(-0.5799, abs=1e-2)

    assert data['y'].mean() == pytest.approx(3.744, abs=1e1)
    assert data['time_cens'].mean() == pytest.approx(0.0, abs=5e-1)
    assert data['deceased'].mean() == pytest.approx(0.8032, abs=1e-1)

    # Test that the model has the correct local parameters
    assert load_protect_model.local_prms == ['Feps']


def test_conditioning_sets(load_protect_model):
    # Test that conditioning sets are correctly set
    conditioning_sets = load_protect_model.conditioning_sets

    assert set(conditioning_sets.keys()) == {
        "no_txy", "no_y", "no_tx",
        "no_proxy1", "no_proxy1_no_y", "no_proxy1_no_tx",
        "no_proxy2", "no_proxy2_no_y", "no_proxy2_no_tx",
        "joint"
    }

    assert set(conditioning_sets["no_txy"]) == {'obs_proxy1', 'obs_proxy2'}
    assert set(conditioning_sets["no_y"]) == {'obs_proxy1', 'obs_proxy2', 'obs_tx'}
    assert set(conditioning_sets["no_tx"]) == {'obs_proxy1', 'obs_proxy2', 'y_enum'}
    assert set(conditioning_sets["no_proxy1"]) == {'obs_proxy2', 'obs_tx', 'obs_y'}
    assert set(conditioning_sets["no_proxy1_no_y"]) == {'obs_proxy2', 'obs_tx'}
    assert set(conditioning_sets["no_proxy1_no_tx"]) == {'obs_proxy2', 'y_enum'}
    assert set(conditioning_sets["no_proxy2"]) == {'obs_proxy1', 'obs_tx', 'obs_y'}
    assert set(conditioning_sets["no_proxy2_no_y"]) == {'obs_proxy1', 'obs_tx'}
    assert set(conditioning_sets["no_proxy2_no_tx"]) == {'obs_proxy1', 'y_enum'}
    assert set(conditioning_sets["joint"]) == {'obs_proxy1', 'obs_proxy2', 'obs_tx', 'obs_y'}

