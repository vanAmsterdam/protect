'''
this file contains the definition of numpyro models
'''

import numpyro
from jax import numpy as jnp, random, vmap
from jax.scipy.special import expit
from numpyro import distributions as dist, sample
from numpyro.handlers import mask, seed, trace, condition
from causalprotect.distributions import PowerGeneralizedWeibullLog as PGW
from causalprotect.utils import create_prior_func, priorspec_to_priorfunc, make_dummy_priorfunc, check_data, time_event_to_time_cens
import numpy as np
from typing import Callable
from pathlib import Path
from warnings import warn
from jax import lax
from functools import partial


# a dict of metadata for each model
model_info_dict = {}
models = {}

class PROTECTModel:
    '''
    a class for a PROTECT model
    '''
    def __init__(self, model: Callable, metadata: dict,
                prior_func: Callable=None, prior_dict: dict=None,
                prior_spec: Path|str=None, prior_varnames: list=None,
                data: dict = None):
        self.model = model
        self.metadata = metadata
        self.model_control = metadata['model_control']
        # check bma
        if 'bma' not in self.model_control:
            self.model_control['bma'] = False

        if self.model_control['bma']:
            assert len(self.model_control['s_bftxs']) == len(self.model_control['s_bfys']), "s_bftxs and s_bftys must be the same length"
            assert len(self.model_control['s_bftxs']) > 1, "s_bftxs and s_bftys must be greater than 1"
        self.f_proxies = metadata['f_proxies']
        self.control_variables = metadata['control_variables']
        self.obs_vars = self.f_proxies + ["y", "tx"]

        # create a default control dictionary
        inference_control = {f"sample_{k}": True for k in self.obs_vars}
        inference_control["enum_tx"] = False
        self.inference_control = inference_control
        self.control = self.model_control | inference_control

        # determine the prior function
        if prior_func is not None:
            self.prior_func = prior_func
        elif prior_dict is not None:
            self.prior_func = create_prior_func(prior_dict)
        elif prior_spec is not None:
            self.prior_func = priorspec_to_priorfunc(prior_spec)
        else:
            print("no priors provided, will create a dummy prior function with Normal(0,5) for all parameters")
            self.prior_func = make_dummy_priorfunc(prior_varnames)

        # make a model with several arguments partialled in
        self.model_with_priors = partial(self.model, prm_fn=self.prior_func)
        self.model_with_priors_and_control = partial(self.model_with_priors, control=self.control)

        # check parameter types
        if 'global_prms' in metadata and 'local_prms' in metadata:
            self.global_prms = metadata['global_prms']
            self.local_prms = metadata['local_prms']
        else:
            if data is None:
                assert 'local_prms' in metadata, "local_prms must be provided in metadata if data is not provided"
                local_prms = metadata['local_prms']
                # get global_prms from a sample of prior_func
                with seed(rng_seed=0):
                    prior_sample = self.prior_func()
                global_prms = [k for k, v in prior_sample.items() if v.shape == ()]
            else:
                # try:
                local_prms, global_prms = self._infer_params(data)
                # except Exception as e:
                    # print(f"Error inferring parameters: {e}")
                    # local_prms, global_prms = [], []
                    # print("could not infer parameters, setting local and global to empty lists, need to update these manually")
                # else:
                    # pass

            if 'global_prms' in metadata:
                set_inferred = set(global_prms)
                set_metadata = set(metadata['global_prms'])
                if set_inferred != set_metadata:
                    raise ValueError(f"mismatch between inferred global parameters {set_inferred} and metadata {set_metadata}")
                self.global_prms = metadata['global_prms']
            else:
                self.global_prms = global_prms
            if 'local_prms' in metadata:
                set_inferred = set(local_prms)
                set_metadata = set(metadata['local_prms'])
                if set_inferred != set_metadata:
                    raise ValueError(f"mismatch between inferred local parameters {set_inferred} and metadata {set_metadata}")
                self.local_prms = metadata['local_prms']
            else:
                self.local_prms = local_prms

        # setup all possible conditioning sets for posterior predictive modes
        proxy_set = [f"obs_{proxy}" for proxy in self.f_proxies]
        conditioning_sets = {
            'no_txy': proxy_set,
            'no_y': proxy_set + ["obs_tx"],
            'no_tx': proxy_set + ["y_enum"],
            'joint': proxy_set + ["obs_tx", "obs_y"]
        }
        for proxy in self.f_proxies:
            obs_proxy = f"obs_{proxy}"
            conditioning_sets[f"no_{proxy}"] = ["obs_tx", "obs_y"] + list(set(proxy_set) - {obs_proxy})
            conditioning_sets[f"no_{proxy}_no_y"] = ["obs_tx"] + list(set(proxy_set) - {obs_proxy})
            conditioning_sets[f"no_{proxy}_no_tx"] = ["y_enum"] + list(set(proxy_set) - {obs_proxy})
        self.conditioning_sets = conditioning_sets


    def _infer_params(self, data, *args, **kwargs):
        '''
        using data, infer the local parameters
        '''
        N = data['time_cens'].shape[0]
        tr = self.get_trace(0, data, *args, **kwargs)
        local_params = []
        global_params = []
        for k, v in tr.items():
            if v['type'] == 'sample' and not v['is_observed']:
                if v['value'].shape == ():
                    global_params.append(k)
                elif v['value'].shape == (N,):
                    local_params.append(k)
                else:
                    warn(f"parameter {k} has shape {v['value'].shape} that is not recognized as local or global")
        
        return local_params, global_params

    def __call__(self, data, control=None, prm_fn=None, obs_masks=None, *args, **kwargs):
        '''
        call the model
        '''
        # PERF: this may be part of scanned functions (e.g. marginalized_hazard_ratio), check jax control flow 
        if control is None:
            control = self.control
        if prm_fn is None:
            prm_fn = self.prior_func
        if 'time_cens' in data and data['time_cens'] is not None:
            if 'N' in control and control['N'] != data['time_cens'].shape[0]:
                raise ValueError(f"control['N'] = {control['N']} does not match data['time_cens'].shape[0] = {data['time_cens'].shape[0]}")
            control['N'] = data['time_cens'].shape[0]
        
        return self.model(data, control, prm_fn, obs_masks, *args, **kwargs)

    def sample(self, rng_seed, *args, **kwargs):
        '''
        sample from the model with a seed
        '''
        with seed(rng_seed=rng_seed):
            return self(*args, **kwargs)


    def prior_sample(self, rng_seed):
        """
        return sample of global parameters from prior
        """
        with seed(rng_seed=rng_seed):
            return self.prior_func()


    def prior_samples(self, rng_seed, nsamples):
        """
        return samples of global parameters from prior
        """
        with seed(rng_seed=rng_seed):
            return vmap(self.prior_sample, in_axes=(0,))(random.split(rng_seed, nsamples))


    def get_trace(self, rng_seed, data, control=None, prm_fn=None, obs_masks=None, *args, **kwargs):
        '''
        get a numpyro.trace of the model
        '''
        if control is None:
            control = self.control
        if prm_fn is None:
            prm_fn = self.prior_func
        control['N'] = data['time_cens'].shape[0]
        return trace(seed(self.model, rng_seed)).get_trace(data, control, prm_fn, obs_masks, *args, **kwargs)
 

    def check_data(self, data, obs_masks=None):
        """
        check if data is compatible with model and metadata
        """
        return check_data(data, self.metadata, obs_masks)


    def model_check_evaluation(self, model_check_result: dict):
        """
        running PROTECTInference.model_checks yields a dictionary of likelihoods for different posterior predictive modes.
        apply the model logic to compare the correct values
        :param model_check_result: a dictionary with structure: {ppmode: {(train/test, obs_site): {'max': [fold1, fold2, ... foldk], 'mean': [...]}}}: likelihood_value}}
        """
        # first, extract only the test likelihoods and sum over folds
        test_likelihoods = {}
        for ppmode, results in model_check_result.items():
            test_lls = {}
            for (split, obs_var), results in results.items():
                if split == 'test':
                    test_lls[obs_var] = results['sum'].sum()
            test_likelihoods[ppmode] = test_lls

        # now, make the comparisons
        # set 1: observation sites should be better than baselines
        # ref: appendix page 12, eqs 5-7
        all_passed = True
        set1 = {}
        for obs_var in self.obs_vars:
            baseline = test_likelihoods['baseline'][obs_var]
            protect = test_likelihoods["no_"+obs_var][obs_var]

            set1[obs_var] = {
                'baseline': baseline,
                'protect': protect,
                'passed': protect > baseline
            }
            all_passed *= set1[obs_var]['passed']

        # set 2: for all proxies, treatment AND outcome should be informative than either one
        # ref: appendix page 12, eqs 8, 9
        set2 = {}
        for proxy in self.f_proxies:
            no_proxy = test_likelihoods["no_"+proxy][proxy]
            no_proxy_no_tx = test_likelihoods["no_"+proxy+"_no_tx"][proxy]
            no_proxy_no_y = test_likelihoods["no_"+proxy+"_no_y"][proxy]
            set2[proxy] = {
                'no_proxy': no_proxy,
                'no_proxy_no_tx': no_proxy_no_tx,
                'no_proxy_no_y': no_proxy_no_y,
                'tx_informative': no_proxy > no_proxy_no_tx,
                'y_informative': no_proxy > no_proxy_no_y,
                'passed': (no_proxy > no_proxy_no_tx) and (no_proxy > no_proxy_no_y)
            }
            all_passed *= set2[proxy]['passed']

        return {
            'set1': set1,
            'set2': set2,
            'all_passed': all_passed
        }

    def simulate_from_prms(self, rng_seed: int, prm_values, nsim: int = 200,
                           keep_deterministic: bool = True,
                           keep_locals: bool = False,
                           maxtime: float = None,
                           control_distributions: dict[str, dist.Distribution] = None,
                           **kwargs) -> dict:
        '''
        simulate data from the model using specific parameters
        :param rng_seed: seed for the random number generator
        :param prm_values: values for global parameters, dictionary with parameter names as keys and values
        :param nsim: number of simulations
        :param keep_deterministic: whether to keep the deterministic sites in the output
        :param maxtime: maximum time for censoring; if left None, it is assumed that the data is not censored
        :param control_distributions: dictionary with control variables and their distributions; if left None, it is assumed all control variables are Normal(0, 1)
        '''

        # check if there are missing control distributions, fill them with Normal(0,1)
        if control_distributions is None:
            control_distributions = {k: dist.Normal(0, 1) for k in self.control_variables}
        else:
            for k in self.control_variables:
                if k not in control_distributions:
                    control_distributions[k] = dist.Normal(0, 1)

        # sample the control variables
        control_data = {}
        for k, d in control_distributions.items():
            rng_seed, k_local = random.split(rng_seed)
            control_data[k] = d.sample(k_local, sample_shape=(nsim,))
        # add the observation sites as placeholders
        control_data.update({k: None for k in self.obs_vars})
        control_data['time_cens'] = None

        # condition the model on the prm_values and get trace by running forward with control_data
        control = self.control | {"N": nsim}
        with seed(rng_seed=rng_seed):
            with condition(data=prm_values):
                tr = trace(self).get_trace(data=control_data, control=control, **kwargs)

        # from the traces grab the observation variables and the deterministic variables
        ## (deterministic variables are deliberately exported by the model)
        trace_keys = list(tr.keys())
        obs_keys   = [f"obs_{k}" for k in self.obs_vars if f"obs_{k}" in trace_keys]
        det_keys   = [k for k in trace_keys if tr[k]['type'] == 'deterministic']
        get_keys   = obs_keys
        if keep_deterministic:
            get_keys += det_keys
        if keep_locals:
            get_keys += self.local_prms
        obs_data   = {k.split('obs_')[-1]: tr[k]['value'] for k in get_keys}

        
        # post-process times by applying censoring (can get rid of extreme values)
        time_cens = time_event_to_time_cens(obs_data['y'], maxtime=maxtime)
        obs_data['deceased'] = time_cens > 0
        obs_data['time_cens'] = time_cens
        if maxtime is not None:
            obs_data['y'] = jnp.abs(time_cens)
        else:
            assert all(jnp.isfinite(obs_data['y']))

        return control_data | obs_data


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
            m = obs_masks['proxy1'] if obs_masks and 'proxy1' in obs_masks else np.ones_like(w).astype(bool)
                
            with mask(mask=m):
                sample('obs_proxy1', dist.Bernoulli(logits=eta), obs=w)

        if control['sample_proxy2']:
            # F -> W part
            eta = Fhat * prms['b_F_proxy2'] - prms['mu_proxy2']

            # sample the proxy
            w = data['proxy2']
            m = obs_masks['proxy2'] if obs_masks and 'proxy2' in obs_masks else np.ones_like(w).astype(bool)
                
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


class PROTECTModelEnsemble(PROTECTModel):
    """
    A class for an ensemble of PROTECT models.
    """
    def __init__(self, models: list[PROTECTModel]):
        """
        Initialize the ensemble with a list of PROTECTModel instances and optional weights.

        :param models: A list of PROTECTModel instances.
        """
        if not models:
            raise ValueError("The ensemble must contain at least one model.")
        if len(models) <= 1:
            raise ValueError("The number of models must be greater than 1.")
        if not all(isinstance(model, PROTECTModel) for model in models):
            raise TypeError("All elements in the models list must be instances of PROTECTModel.")

        self.models = models

        # create ensemble model
        ensemble_model = _create_ensemble_model([model.model_with_priors_and_control for model in models])

        # Use metadata from the first model as the ensemble's metadata
        super().__init__(
            model=ensemble_model,  # Placeholder, as the ensemble doesn't have a single model
            metadata=models[0].metadata | {'local_prms': models[0].local_prms, 'global_prms': models[0].global_prms},
            prior_func=models[0].prior_func,
            prior_dict=models[0].metadata.get("prior_dict"),
            prior_spec=models[0].metadata.get("prior_spec"),
            prior_varnames=models[0].metadata.get("prior_varnames"),
        )



def _create_ensemble_model(models: list[Callable]) -> PROTECTModelEnsemble:
    n_models = len(models)

    def ensemble_model(data=None, *args, **kwargs):

        # sample model weights from Dirichlet distribution
        model_weights = sample('model_weights', dist.Dirichlet(jnp.ones(n_models)))

        # sample the model for this sample
        model_idx = sample('model_idx', dist.Categorical(model_weights))

        # apply the model
        return lax.switch(model_idx, models, data)

    # return the ensemble model
    return ensemble_model
