'''
define the statistical model
'''

import numpy as np, numpyro
from numpyro import distributions as dist, sample
from numpyro.handlers import mask
from causalprotect.distributions import PowerGeneralizedWeibullLog as PGW, NegativeHalfNormal

from jax.scipy.special import expit

def tutorial_model(data, control, prm_fn, obs_masks=None):
    '''
    a probabilistic model for predicting generating linear predictors following the DAG), using only F as latent factor and other covariates to directly condition on 
    :param data dict: a dictionary with jnp.ndarrays for all observed data that is used to condition on, and None for unobserved / to marginalize out data.
    :param control dict: a dicitonary with control arguments, sample_globals, N, dimensions for everything etc....
    :param prm_fn function: returns a dictionary with parameters, either sampled from priors or fixed numbers (e.g. in posterior predictions)
    :param obs_masks dict: a dictionary with binary arrays of the same size as the data with indicators for observations
    :returns: a jnp.array that is the linear predictor for a glm or survival model
    '''
    # get global parameters
    prms = prm_fn()

    # handle potential bayesian model averaging with different priors (hyperparameters)
    if control['bma']:
        s_bftxs = control['s_bftxs']
        s_bfys = control['s_bfys']
        n_models = s_bftxs.shape[0]

        # sample model weights from Dirichlet distribution
        model_weights = sample('model_weight', dist.Dirichlet(np.ones(n_models)))

        # sample the model for this sample
        model_idx = sample('model_idx', dist.Categorical(model_weights))

        # setup the prior
        s_bftx = s_bftxs[model_idx]
        s_bfy = s_bfys[model_idx]

        # sample the model parameters
        prms['b_F_tx'] = sample('b_F_tx', dist.HalfNormal(s_bftx))
        prms['b_F_y'] = sample('b_F_y', NegativeHalfNormal(s_bfy))

    # data plate
    with numpyro.plate('obs', control['N']):
        mu_Fhat = data['age'] * prms['b_age_F']
        # sample residual for F (gives better sampling behavior than directly sampling with conditional mean)
        Feps = sample('Feps', dist.Normal())
        Fhat = mu_Fhat + Feps

        # pull Fhat through sigmoid transformation for fixing scale
        if control['Fsigmoid']:
            Fhat = expit(Fhat)
        # store Fhat in the samples
        numpyro.deterministic('Fhat', Fhat)

        # run proxy models
        if control['sample_proxy1']:
            # F -> W part
            eta = Fhat * prms['b_F_proxy1'] - prms['mu_proxy1']

            # sample the proxy
            w = data['proxy1']
            m = obs_masks['proxy1'] if obs_masks and 'proxy1' in obs_masks else np.ones((control['N'],)).astype(bool)
                
            with mask(mask=m):
                sample('obs_proxy1', dist.Bernoulli(logits=eta), obs=w)

        if control['sample_proxy2']:
            # F -> W part
            eta = Fhat * prms['b_F_proxy2'] - prms['mu_proxy2']

            # sample the proxy
            w = data['proxy2']
            m = obs_masks['proxy2'] if obs_masks and 'proxy2' in obs_masks else np.ones((control['N'],)).astype(bool)
                
            with mask(mask=m):
                sample('obs_proxy2', dist.Bernoulli(logits=eta), obs=w)

        # treatment model
        eta = Fhat * prms['b_F_tx'] - prms['mu_tx']

        # observe tx, enumerate out, or marginalize out without enumeration
        obstx    = data['tx'] if control['sample_tx'] else None
        infer_tx = {'enumerate': 'parallel'} if control['enum_tx'] else None
        txhat    = sample('obs_tx', dist.Bernoulli(logits=eta), obs=obstx, infer=infer_tx)

        # store eta tx as a deterministic site to be able to sample values later
        numpyro.deterministic('eta_tx', eta)

        # outcome model for the linear predictor
        ## also keep track of the no-treatment and with-treatment lps
        lps_notx  = Fhat * prms['b_F_y']
        lps = lps_notx + txhat * prms['b_tx_y']
        lps_dotx = lps_notx + prms['b_tx_y']

        # possibly add treatment interactions
        b_tx_y_marginal = prms['b_tx_y']
        
        if control['Ftx_int']:
            lps += txhat * Fhat * prms['b_Ftx_y']
            lps_dotx += Fhat * prms['b_Ftx_y']
            b_tx_y_marginal += Fhat.mean(axis=-1) * prms['b_Ftx_y']

        # log the marginal treatment effect when treatment interactions are modelled
        numpyro.deterministic('b_tx_y_marginal', b_tx_y_marginal)

        # log some deterministic sites
        numpyro.deterministic('lp', lps)
        numpyro.deterministic('lp_notx', lps_notx)
        numpyro.deterministic('lp_dotx', lps_dotx)

        return_value = lps
        
        # sample outcome
        if control['sample_y']:
            betas  = prms['beta0'] + lps

            y = data['time_cens']
            # m = obs_masks['obs_y'] if obs_masks and 'obs_y' in obs_masks else np.ones_like(y).astype(bool)
            # with mask(mask=m):
            return_value = sample('obs_y', PGW(0.0, betas, prms['alpha0'], prms['nu0']), obs=y)

        return return_value
