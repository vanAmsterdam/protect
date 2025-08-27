import numpyro
import pytest
from jax import numpy as jnp, vmap

from causalprotect.models import PROTECTModel
from causalprotect.inference import PROTECTInference
from numpyro import distributions as dist, sample

def mixture_model(data, control, prm_fn, obs_masks=None):

    prms = prm_fn()

    with numpyro.plate('obs', control['N']):
        Fhat = sample('Feps', dist.Normal())

        # # sample proxy
        if control['sample_w']:
            mu_w = Fhat * prms['b_F_w'] - prms['mu_w']
            sample('obs_w', dist.Normal(mu_w, 1.0), obs=data['w'])

        eta = Fhat * prms['b_F_tx'] - prms['mu_tx']
        # observe tx, enumerate out, or marginalize out without enumeration
        obstx    = data['tx'] if control['sample_tx'] else None
        infer_tx = {'enumerate': 'parallel'} if control['enum_tx'] else None
        txhat    = sample('obs_tx', dist.Bernoulli(logits=eta), obs=obstx, infer=infer_tx)

        # sample outcome
        mu = Fhat * prms['b_F_y'] + txhat * prms['b_tx_y'] - prms['mu_y']

        # keep track of mu
        numpyro.deterministic('mu', mu)

        return_value = mu
        
        # sample outcome
        if control['sample_y']:
            return_value = sample('obs_y', dist.Normal(mu, prms['sigma']), obs=data['y'])

        return return_value

metadata = {
    "model_control": {},
    "f_proxies": ['w'],
    "control_variables": [],
    "local_prms": ["Feps"]
}
prior_varnames = ['b_F_w', 'mu_w',
                 'b_F_tx', 'mu_tx',
                 'b_F_y', 'mu_y', 'b_tx_y',
                 'sigma']

model = PROTECTModel(mixture_model, metadata, prior_varnames=prior_varnames)

# setup global parameter values
prms10 = {'b_F_w': 1.0, 'mu_w': 0.0,
            'b_F_tx': 0.0, 'mu_tx': 0.0,
            'b_F_y': 0.0, 'mu_y': 0.0, 'b_tx_y': 0.0,
            'sigma': 1.0}
prms11 = {'b_F_w': 1.0, 'mu_w': 0.0,
            'b_F_tx': 0.0, 'mu_tx': 0.0,
            'b_F_y': 1.0, 'mu_y': 0.0, 'b_tx_y': 0.0,
            'sigma': 1.0}
prms21 = {'b_F_w': 2.0, 'mu_w': 0.0,
            'b_F_tx': 0.0, 'mu_tx': 0.0,
            'b_F_y': 1.0, 'mu_y': 0.0, 'b_tx_y': 0.0,
            'sigma': 1.0}
prms22 = {'b_F_w': 2.0, 'mu_w': 0.0,
            'b_F_tx': 0.0, 'mu_tx': 0.0,
            'b_F_y': 2.0, 'mu_y': 0.0, 'b_tx_y': 0.0,
            'sigma': 1.0}

prms = {k: jnp.array([prms10[k], prms11[k], prms21[k], prms22[k]]) for k in prms10}

# setup observed data

data = {
    'w': jnp.concat([jnp.zeros(5, ), jnp.ones(5, )]),
    'tx': jnp.zeros((10, )),
    'y': jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0,
                    -2.0, -1.0, 0.0, 1.0, 2.0]),
}
data['time_cens'] = data['y']
data['deceased'] = jnp.ones((10, ))
num_obs = int(10)

protector = PROTECTInference(model, data, metadata, check_data=False)

num_global_smps = prms['b_F_w'].shape[0]

def test_inverse_conditional_expectations():
    """
    Test that the inverse conditional expectations are correct.
    """

    # helper functions for the following:
    ## P(F | W=w) = N(mu_F|w, sigma_F|w)

    def _mu_F_given_w(w, b_F_w, mu_w):
        return (b_F_w / (1 + b_F_w**2)) * (w - mu_w)
    def _sigma_F_given_w(b_F_w):
        return jnp.sqrt(1 / (1 + b_F_w**2))
    
    # calculate mu_F for all ws
    def _mu_F(b_F_w, mu_w):
        return vmap(lambda w: _mu_F_given_w(w, b_F_w, mu_w))(data['w'])

    mu_F = vmap(_mu_F, in_axes=(0, 0))(prms['b_F_w'], prms['mu_w'])

    # calculate analytical expressions for mu_y
    b_F_y = jnp.expand_dims(prms['b_F_y'], -1)
    mu_y = b_F_y * mu_F

    # given known analytical distributions of F given W and parameter, we can compute expectations for mu_y 
    lls, sites = protector.run_postpred_grid(prms, deterministic_sites=['mu'])
    mu_grid = sites['no_txy']['mu']

    mu_y == pytest.approx(mu_grid, abs=0.001)
