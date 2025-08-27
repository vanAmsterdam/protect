"""
running protect inference
"""

from functools import partial

import arviz as az
import numpy as np
from jax import numpy as jnp
from jax import random, vmap, pmap, lax
from jax.lax import scan
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro import distributions as dist
from numpyro.infer.hmc import NUTS
from numpyro.infer.mcmc import MCMC, MCMCKernel
from numpyro.infer.util import log_likelihood
from numpy.polynomial.hermite_e import hermegauss

from causalprotect.models import PROTECTModel
from causalprotect.distributions import PowerGeneralizedWeibullLog as PGW
from causalprotect.utils import optimize_pgw, time_event_to_time_cens, generate_cv_intrain_matrix, summarize_likelihoods
from causalprotect.utils import get_log_likelihoods_from_trace
from causalprotect.utils import harrell_c_streaming, roc_auc


class PROTECTInference:
    """
    PROTECT inference class
    """

    def __init__(self, protect_model: PROTECTModel, data: dict, obs_masks: dict=None, check_data=True, maxtime=10):
        """
        initialize the PROTECT inference class
        protect_model: PROTECTModel object
        data: dictionary with the data
        obs_masks: dictionary with observation masks
        check_data: check if data is concordant with the model
        maxtime: maximum time to censor
        """
        self.protect_model = protect_model
        self.data = data
        self.maxtime = maxtime

        self.inference_control = protect_model.inference_control
        self.inference_control["N"] = data["time_cens"].shape[0]
        self.default_control = protect_model.model_control | self.inference_control

        if check_data:
            protect_model.check_data(data, obs_masks)

        # TODO: implement check for user provided obs_masks

        # user provided observation masks
        if obs_masks is not None:
            self.obs_masks = obs_masks  
        # get observation masks based on whether data are finite
        else:
            # TODO: think if optionally having None as self.obs_masks would work
            self.obs_masks = {k: ~np.isnan(v) for k, v in data.items()}
            # if any(np.isnan(v).any() for v in data.values()):
                # self.obs_masks = {k: ~np.isnan(v) for k, v in data.items()}
            # else:
                # self.obs_masks = None

        # prepare some placeholders
        self.pp_mcmcs = {} 
        self.mcmc_samples = {}

        # set fixed arguments to simplify model call
        self.mcmc_model = partial(
            protect_model.model,
            prm_fn=protect_model.prior_func,
        )

        self.pp_model_template = partial(
            protect_model.model,
            prm_fn=protect_model.prior_func,
            data=self.data,
            obs_masks=self.obs_masks,
        )

        self.mcmc_model_with_data = partial(
            protect_model.model,
            prm_fn=protect_model.prior_func,
            data=self.data,
            obs_masks=self.obs_masks,
        )

        pp_modes = ["no_y", "no_tx", "no_txy"]
        for proxy in protect_model.f_proxies:
            pp_modes.append("no_" + proxy)
            pp_modes.append("no_" + proxy + "_no_y")
            pp_modes.append("no_" + proxy + "_no_tx")
        self.pp_modes = pp_modes

    def run_inference(
        self,
        rng_key,
        data=None,
        obs_masks=None,
        *args,
        sampler: MCMCKernel = NUTS,
        mcmc_kwargs: dict = {},
        save_samples=True,
        **kwargs,
    ):
        """
        run MCMC inference
        *args and **kwargs go to model call
        """
        if data is None:
            data = self.data
        if obs_masks is None:
            obs_masks = self.obs_masks
        control = self.default_control.copy()
        control["N"] = data["time_cens"].shape[0]

        # update mcmc_kwargs with default values only if they do not occur in mcmc_kwargs
        default_mcmc_kwargs = {"num_warmup": 1000, "num_samples": 1000, "num_chains": 4} 
        for key, value in default_mcmc_kwargs.items():
            mcmc_kwargs.setdefault(key, value)

        # if not hasattr(self, "mcmc"):
            # self.mcmc = MCMC(sampler(self.mcmc_model), **mcmc_kwargs)
        # TODO: can we make mcmc object persistent? make an underscore function that takes the object as an argument?
        model = partial(self.mcmc_model, control=control)
        self.mcmc = MCMC(sampler(model), **mcmc_kwargs)

        samples = self._run_inference(rng_key, self.mcmc, data, obs_masks, *args, **kwargs)
        if save_samples:
            self.mcmc_samples["posterior"] = samples
        self.train_data = data
        self.train_obs_masks = obs_masks
        return samples


    def _run_inference(self, rng_key, mcmc, data, obs_masks, *args, **kwargs):
        """
        run MCMC inference, state-less version
        """
        mcmc.run(rng_key, data=data, obs_masks=obs_masks, *args, **kwargs)
        return mcmc.get_samples(group_by_chain=False)


    def get_samples(self, group_by_chain=True):
        """
        get samples from MCMC
        """
        return self.mcmc.get_samples(group_by_chain=group_by_chain)

    def get_summary(self, additional_vars=[]):
        """
        summarize MCMC samples of global parameters
        additional_vars: list of additional variables to summarize, e.g. deterministic global sites like b_tx_y_marginal
        """
        if not hasattr(self, "mcmc"):
            raise ValueError("No MCMC samples found. First run_inference()")

        if not hasattr(self, "azd"):
            self.azd = az.from_dict(self.mcmc.get_samples(group_by_chain=True))
        
        summary_vars = self.protect_model.global_prms + additional_vars

        return az.summary(self.azd, summary_vars, hdi_prob=0.95)

    def run_postpred_modes(
        self,
        rng_key,
        pp_modes: list = None,
        num_local_draws: int = 100,
        group_by_chain=False,
        num_workers: int = 1,
        sampler: MCMCKernel = NUTS,
        mcmc_kwargs: dict = {},
        verbose=False,
        *args,
        **kwargs,
    ):
        """
        run posterior predictive checks
        num_local_draws: number of draws of local parameter for each value of global parameters
        num_workers: number of workers to use for parallel computation
        """
        if pp_modes is None:
            pp_modes = self.pp_modes
        # print("WARNING: subsetting pp modes for testing")
        # pp_modes = pp_modes[:2]

        rng_keys = random.split(rng_key, len(pp_modes))
        pp_samples = {}
        for ppmode, key in zip(pp_modes, rng_keys):
            if verbose:
                print(f"Running posterior predictive mode: {ppmode}")
            pp_samples[ppmode] = self.run_postpred_mcmc(
                rng_key,
                ppmode=ppmode,
                num_local_draws=num_local_draws,
                group_by_chain=group_by_chain,
                num_workers=num_workers,
                sampler=sampler,
                mcmc_kwargs=mcmc_kwargs,
                *args,
                **kwargs,
            )
        return pp_samples

    def run_postpred_mcmc(
        self,
        rng_key,
        posterior_samples, 
        pp_mode="no_y",
        data=None,
        obs_masks=None,
        num_local_draws: int = 100,
        group_by_chain=False,
        num_workers: int = 1,
        sampler: MCMCKernel = NUTS,
        mcmc_kwargs: dict = {},
        global_sample_shape = None,
        return_lls=False,
        *args,
        **kwargs,
    ):
        """
        run posterior predictive checks
        posterior_samples: samples from the posterior
        num_local_draws: number of draws of local parameter for each value of global parameters
        num_workers: number of workers to use for parallel computation
        """
        global_samples = {k: v for k, v in posterior_samples.items() if k in self.protect_model.global_prms}
        if global_sample_shape is None:
            global_sample_shape = global_samples[self.protect_model.global_prms[0]].shape
        sliced_samples = _slice_posterior_for_pp(
            global_samples, global_sample_shape=global_sample_shape, 
            num_samples_out=num_local_draws, group_by_chain=group_by_chain
        )

        global_sample0 = {k: v[0] for k, v in global_samples.items()}
        if group_by_chain:
            global_sample0 = {k: v[0] for k, v in global_sample0.items()}
        
        if data is None:
            data = self.data
        if obs_masks is None:
            obs_masks = self.obs_masks

        if return_lls:
            return_value = 'both'
        else:
            return_value = 'samples'

        pp_fun = _make_mcmc_postpred_fn(
            global_sample0, pp_mode, self.mcmc_model, data, self.default_control, obs_masks=obs_masks, return_value=return_value, mcmc_kwargs=mcmc_kwargs
        )

        if num_workers > 1:
            raise NotImplementedError(
                "parallel not implemented, think about how to copy the mcmc object to different workers"
            )
        else:
            if group_by_chain:
                num_chains = global_sample_shape[0]
                if return_lls:
                    locals_samples, lls = vmap(pp_fun)(random.split(rng_key, num_chains), sliced_samples)
                else:
                    locals_samples = vmap(pp_fun)(random.split(rng_key, num_chains), sliced_samples)
            else:
                if return_lls:
                    locals_samples, lls = pp_fun(rng_key, sliced_samples)
                else:
                    locals_samples = pp_fun(rng_key, sliced_samples)

        outsamples = sliced_samples | locals_samples
        self.mcmc_samples[pp_mode] = outsamples
        
        if return_lls:
            return outsamples, lls
        else:
            return outsamples


    def slice_posterior_for_pp(self, samples, num_samples_out=100, group_by_chain=False, drop_locals=True):
        """
        slice posterior samples for running posterior predictions where mcmc is required
        :param samples: posterior samples to slice
        :param num_samples_out: number of global parameters to get
        :param group_by_chain: group samples by chain?
        :param drop_locals: drop local parameters from the samples?
        """
        global_prms = self.protect_model.global_prms

        # prepare the samples by slicing them
        global_sample_shape = samples[global_prms[0]].shape
        
        sliced_samples = _slice_posterior_for_pp(
            samples, global_sample_shape, num_samples_out=num_samples_out, group_by_chain=group_by_chain
        )

        if drop_locals:
            # drop the non-global prms
            sliced_samples = {k: v for k, v in sliced_samples.items() if k in global_prms}

        return sliced_samples
    

    def calculate_log_likelihoods(self,
                                   samples,
                                   model=None,
                                   data=None, control=None, obs_masks=None, group_by_chain=False,
                                   ppmode = "posterior",
                                   **kwargs):
        """
        get log likelihoods of the model, per observation site, sample and patient
        data: optional argument (e.g. when checking log likelihood of new data, note that the samples contain local variables that typically condition on the data, so the data should match the data used for the samples)
        group_by_chain: are samples grouped by chain?
        ppmode: optional argument specifying the posterior predictive mode that was used to generate the samples
        **kwargs: these go to the model call
        return: dictionary of log likelihoods of the model per observation site (e.g. obs_tx, obs_proxy1, obs_proxy2, etc.), per chain, per sample, per patient
        """
        if control is None:
            control = self.default_control.copy()
        if data is None:
            if ppmode == "posterior":
                data = self.train_data
            else:
                data = self.data
        if model is None:
            model = self.protect_model.model

        control["N"] = data["time_cens"].shape[0]
        if obs_masks is None:
            obs_masks = self.obs_masks

        batch_ndims = 2 if group_by_chain else 1

        log_lik = log_likelihood(
            self.protect_model.model, samples, batch_ndims=batch_ndims, data=data, control=control, prm_fn = self.protect_model.prior_func, obs_masks=obs_masks, **kwargs
        )

        return log_lik

    def log_likelihood_per_patient_for_all_ppmodes(self):
        """
        get log likelihoods of the model per patient for all ppmodes, convenience function
        """
        if len(self.mcmc_samples) == 0:
            raise ValueError("No MCMC samples found. First run_inference()")

        lls_out = {}

        data = self.train_data
        control = self.default_control
        control["N"] = data["time_cens"].shape[0]
        model = partial(self.protect_model.model,
                        prm_fn=self.protect_model.prior_func,
                        data=data,
                        obs_masks=self.train_obs_masks,
                        control=control)


        for ppmode, samples in self.mcmc_samples.items():
            if ppmode == "posterior":
                continue
            if "no_tx" in ppmode:
                print(f"Warning: log likelihood for y not implemented yet when enumerating treatment, ppmode = {ppmode}")

            lls = log_likelihood(
                model, samples, batch_ndims=1 
            )

            # calculate likelihoods per observation site and patient
            lls_out[ppmode] = _log_likelihood_per_patient(lls)

        return lls_out


    def model_checks(self, rng_key, num_folds=5,
                     do_baseline=True,
                     do_postpred=True,
                     verbose=False,
                     inference_mcmc_kwargs={},
                     num_global_samples=1000,
                     baseline_mcmc_kwargs={},
                     pp_inference='grid',
                     postpred_mcmc_kwargs={},
                     grid_kwargs={},
                     global_samples=None,
                     *args, **kwargs
                     ):
        """
        run model checks
        this means cross-validating this procedure:
        - run inference on train slice
        - run all posterior predictive modes on test slice
        - calculate log likelihoods
        - summarize log likelihoods
        :param rng_key: random key for mcmc
        :param num_folds: number of folds for cross-validation
        :param do_baseline: do baseline inference?
        :param do_postpred: do posterior predictive inference? either a logical or a list of ppmodes
        :param verbose: print progress?
        :param inference_mcmc_kwargs: kwargs for inference mcmc
        :param num_global_samples: for how many global samples to run posterior predictive checks
        :param baseline_mcmc_kwargs: kwargs for baseline mcmc
        :param postpred_mcmc_kwargs: kwargs for posterior predictive mcmc
        :param grid_kwargs: kwargs for grid-based posterior predictive checks
        :param *args: additional arguments for the model call
        :param **kwargs: additional keyword arguments for the model call
        """

        # prepare rng keys for reproducibility
        rng_folds, rng_baseline, rng_postpred = random.split(rng_key, 3)

        # generate random indices for the folds
        N_total = self.data["time_cens"].shape[0]
        in_train_mat = generate_cv_intrain_matrix(rng_folds, N_total, num_folds)
        N_test = N_total// num_folds
        N_train = N_total - N_test

        # prepare mcmc object for inference and posterior predictives
        full_data_model = partial(self.mcmc_model, control=self.default_control)
        inference_control = self.default_control | {"N": N_train}
        inference_model = partial(self.mcmc_model, control=inference_control)
        # update mcmc_kwargs with default values only if they do not occur in mcmc_kwargs
        default_inference_mcmc_kwargs = {
            "num_warmup": 500,
            "num_samples": 1500,
            "num_chains": 1,
            "progress_bar": False,
            "jit_model_args": True,
            "chain_method": "sequential" # NOTE <- this throws a warning when later using an outer pmap as we're nesting pmaps
        }
        for key, value in default_inference_mcmc_kwargs.items():
            inference_mcmc_kwargs.setdefault(key, value)
        inference_mcmc = MCMC(NUTS(inference_model), **inference_mcmc_kwargs)

        # setup model for likelihood calculation
        ll_control = self.default_control | {"N": N_total}
        ll_model = partial(
            self.mcmc_model,
            control=ll_control,
            data=self.data,
            obs_masks=self.obs_masks,
        )

        global_prm_names = self.protect_model.global_prms
        global_sample_shape = (inference_mcmc_kwargs["num_chains"] * inference_mcmc_kwargs["num_samples"],)

        # helper function for fetching the train data and obs masks
        def _get_train_data_and_obs_masks(in_train):
            # get train data and obs masks
            train_iis = jnp.where(in_train, size=N_train)[0]
            train_data = {k: jnp.take(v, train_iis) for k, v in self.data.items()}
            train_obs_masks = {k: jnp.take(v, train_iis) for k, v in self.obs_masks.items()}
            return train_data, train_obs_masks

        # start doing the actual work
        ll_summaries = {}
        if do_baseline:
            # run baseline inference on the train data
            if verbose:
                print("running baseline inference on train folds")

            baseline_fn_base = _make_baseline_fn(
                global_prm_names,
                inference_model,
                inference_control,
                mcmc_kwargs=baseline_mcmc_kwargs
            )

            def baseline_fn(rng_key, in_train):
                train_data, train_obs_masks = _get_train_data_and_obs_masks(in_train)

                # run baseline inference on train data
                return baseline_fn_base(rng_key, train_data, train_obs_masks)

            keys_baseline = random.split(rng_baseline, num_folds)
            baseline_samples = pmap(baseline_fn)(keys_baseline, in_train_mat)
            baseline_lls = log_likelihood(ll_model, baseline_samples, batch_ndims=2)
            baseline_lls_per_patient = vmap(_log_likelihood_per_patient)(baseline_lls)
            ll_summary_baseline = vmap(partial(summarize_likelihoods, obs_masks=self.obs_masks))(baseline_lls_per_patient, in_test=~in_train_mat)
            ll_summaries['baseline'] = ll_summary_baseline
        
        if do_postpred or (isinstance(do_postpred, list) and len(do_postpred) > 0):
            if global_samples is None:
                def run_inference_on_fold(rng_key, in_train):
                    train_data, train_obs_masks = _get_train_data_and_obs_masks(in_train)
                    rng_train, rng_carry = random.split(rng_key)

                    # run inference on train data
                    # TODO: remove this underscore method that does almost nothing
                    posterior_samples = self._run_inference(
                        rng_train, inference_mcmc, data=train_data, obs_masks=train_obs_masks,
                        *args, **kwargs
                    )
                    sliced_samples = _slice_posterior_for_pp(
                        posterior_samples, global_sample_shape=global_sample_shape, 
                        num_samples_out=num_global_samples, group_by_chain=False
                    )
                    sliced_samples = {k: v for k, v in sliced_samples.items() if k in global_prm_names}
                    return rng_carry, sliced_samples

                # run the actual inference on the training fold
                if verbose:
                    print("running inference on train folds")
                rng_key, post_samples = scan(run_inference_on_fold, rng_postpred, in_train_mat)

                # get only global samples from posterior
                global_samples = {k: v for k, v in post_samples.items() if k in global_prm_names}

            if pp_inference == "grid":
                # setup postpred with grid
                # set parameters for the grid-based posterior predictive checks
                grid_kwargs_defaults = {
                    'K': 32,
                    'accelerator': 'scan',
                }
                for key, value in grid_kwargs_defaults.items():
                    grid_kwargs.setdefault(key, value)
                ## by default, use gauss hermite quadrature with 32 points,
                ## otherwise, use user-defined grid and weights
                if "local_prm_values" in grid_kwargs:
                    local_prm_values = grid_kwargs["local_prm_values"]
                    if "local_prm_value_weights" in grid_kwargs:
                        local_prm_value_weights = grid_kwargs["local_prm_value_weights"]
                    else:
                        local_prm_value_weights = jnp.ones_like(local_prm_values)
                elif "local_value_min" in grid_kwargs and "n_local_values" in grid_kwargs:
                    # make a grid of local parameter values, 
                    # starting at lcoal_value_min and ending at local_value_max
                    # equally spaced in cdf space
                    xs_u = jnp.linspace(dist.Normal(0, 1).cdf(grid_kwargs['local_value_min']),
                                        dist.Normal(0, 1).cdf(-1 * grid_kwargs['local_value_min']),
                                        grid_kwargs['n_local_values'])
                    local_prm_values = jnp.clip(dist.Normal(0, 1).icdf(xs_u), 
                                                min = grid_kwargs['local_value_min'],
                                                max = -1 * grid_kwargs['local_value_min'])
                    local_prm_value_weights = jnp.ones_like(local_prm_values)
                else:
                    local_prm_values, local_prm_value_weights = hermegauss(grid_kwargs["K"])

                # run the posterior predictive checks
                # create function that runs posterior predictive checks for a single global sample
                pp_fun_for_sample = partial(_grid_postpred_for_sample,
                                local_prm_name = self.protect_model.local_prms[0],
                                local_prm_values = local_prm_values,
                                model = full_data_model,
                                data = self.data,
                                conditioning_sets = self.protect_model.conditioning_sets,
                                local_prm_value_weights = local_prm_value_weights,
                                )

                # do nested vmap: for each fold, for each global sample
                if verbose:
                    print("running grid-based posterior predictive checks for all folds")
                if grid_kwargs['accelerator'] == "vmap":
                    lls, site_values = vmap(vmap(pp_fun_for_sample))(global_samples)
                elif grid_kwargs['accelerator'] == "scan":
                    def scannable_fn(carry, x):
                        return carry, pp_fun_for_sample(x)
                    def scan_fold(x):
                        _, y = scan(scannable_fn, None, x)
                        return y
                    lls, site_values = vmap(scan_fold)(global_samples)

                # summarize likelihoods per patient
                for setname, lls_set in lls.items():
                    # lls_set shape is (num_folds, num_global_samples, num_patients)
                    lls_per_patient = vmap(_log_likelihood_per_patient)(lls_set)
                    ll_summary = vmap(partial(summarize_likelihoods, obs_masks=self.obs_masks))(lls_per_patient, in_test=~in_train_mat)
                    ll_summaries[setname] = ll_summary

                return ll_summaries, in_train_mat

            elif pp_inference == "mcmc":

                # setup posterior predictive with mcmc
                # check if do all pp_modes or a subset
                if isinstance(do_postpred, list):
                    pp_modes = do_postpred
                else:
                    pp_modes = self.pp_modes
                
                # extract the first post_sample from the first fold
                global_sample0 = {k: v[0, 0] for k, v in global_samples.items()}

                pp_funs = {}
                for pp_mode in pp_modes:
                    pp_funs[pp_mode] = _make_mcmc_postpred_fn(
                        global_sample0, pp_mode, self.mcmc_model, self.data, self.default_control, obs_masks=self.obs_masks, mcmc_kwargs=postpred_mcmc_kwargs
                    )

                pp_keys = random.split(rng_key, num_folds)
                ll_summaries = {}
                lls_per_patient = {}
                for ppmode, pp_fun in pp_funs.items():
                    # print(f"running posterior predictive mode: {ppmode}")
                    lls = pmap(pp_fun)(pp_keys, global_samples)
                    lls_per_patient[ppmode] = lls
                    ll_summary = vmap(partial(summarize_likelihoods, obs_masks=self.obs_masks))(lls, in_test=~in_train_mat)
                    ll_summaries[ppmode] = ll_summary

            else:
                raise ValueError(f"Unknown posterior predictive inference method: {pp_inference}")

        return ll_summaries, in_train_mat


    def infer_baselines(self,
                        rng_key,
                        mcmc_kwargs = {}
                        ):
        """
        infer the baseline models for all observation nodes;
        these models only condition on direct parents of the observation nodes in the DAG, not on the latent factor
        used in Eqs 5-7 of the appendix of https://doi.org/10.1038/s41598-022-09775-9 
        :param rng_key: random key for mcmc
        :param mcmc_kwargs: kwargs for mcmc
        """

        # make a function that runs mcmc on the model with the latent factor set to zero
        baseline_fn = _make_baseline_fn(
            self.protect_model.global_prms,
            self.pp_model_template,
            self.default_control,
            mcmc_kwargs=mcmc_kwargs
        )

        samples = baseline_fn(rng_key, self.data, self.obs_masks)
        self.mcmc_samples["baseline"] = samples
        
        # calculate the log likelihoods
        ## nb use the "pp_model_template" to calculate lls, otherwise it'll calculate lls for F_params as well as these are now 'is_observed=True' due to the conditioning
        # TODO: refactor the likelihood calculations
        # flat_model_substitute = handlers.substitute(self.pp_model_template, data=F_params)
        lls = log_likelihood(self.pp_model_template, samples, control=self.default_control)
        lls_per_patient = _log_likelihood_per_patient(lls)

        return samples, lls_per_patient


    def get_marginalized_hazard_ratio(self, rng_key, no_txy_samples=None, posterior_samples=None, data=None, obs_masks=None, maxtime=10, max_retries=5):
        """
        calculate the marginalized hazard ratio, potentially on new data
        """

        # get samples from the no_txy posterior predictive mode
        # this simulates the setting in an rct where treatment status and outcome are unobserved
        if no_txy_samples is None:
            if "no_txy" in self.mcmc_samples:
                no_txy_samples = self.mcmc_samples["no_txy"]
            else:
                print("getting no_txy samples from posterior predictive")
                if posterior_samples is None:
                    raise ValueError("no_txy_samples or posterior_samples must be provided")
                no_txy_samples = self.run_postpred_mcmc(
                    rng_key,
                    posterior_samples=posterior_samples,
                    ppmode="no_txy",
                    num_local_draws=500,
                    data=data,
                    obs_masks=obs_masks,
                    group_by_chain=False,
                )
        else:
            if data is not None:
                raise "when supplying no_txy_samples, data is ignored"
            if obs_masks is not None:
                raise "when supplying no_txy_samples, obs_masks is ignored"

        num_obs = no_txy_samples['Fhat'].shape[-1]

        # prep the function
        start = jnp.array([
            jnp.mean(no_txy_samples['beta0']),
            jnp.mean(no_txy_samples['b_tx_y']),
            jnp.mean(no_txy_samples['alpha0']),
            jnp.mean(no_txy_samples['nu0']),
        ])
        carry0 = dict(i=jnp.array(0, jnp.int32), 
                      params=start,
                      converged=jnp.array(False, jnp.bool_),
                      time0=jnp.zeros(num_obs),
                      time1=jnp.zeros(num_obs),
                      )

        def run_until_converged(rng_key, sample):
            """
            run the marginalization until convergence or max_retries
            """
            # carry is a dictionary with the rng_key, iteration index, parameters and convergence status
            carry = carry0 | {'rng_local': rng_key, 'rng_carry': rng_key}

            # define the condition and body functions for scan
            def cond_fn(carry):
                return jnp.logical_not(carry['converged']) & (carry['i'] < max_retries)

            def body_fn(carry):
                rng_carry, rng_local = random.split(carry['rng_carry'])
                result, t0, t1 = _marginalize_hazard_ratio_pgw(rng_local, sample, maxtime=maxtime, start=start)
                carry['params'] = result.position
                carry['converged'] = result.converged
                carry['i'] += 1
                carry['rng_local'] = rng_local
                carry['rng_carry'] = rng_carry
                carry['time0'] = t0
                carry['time1'] = t1
                return carry
            

            result = lax.while_loop(cond_fn, body_fn, carry)

            return result['rng_carry'], result

        _, results = scan(run_until_converged, rng_key, no_txy_samples)

        return results
    
    def run_postpred_grid(self, 
                          global_samples,
                          deterministic_sites = [],
                          grid_kwargs = {},
                          ):
        """
        perform posterior predictive checks using distcretized prior samples from local parameter
        """

        grid_kwargs_defaults = {
            'K': 32,
            'accelerator': 'scan',
        }
        for key, value in grid_kwargs_defaults.items():
            grid_kwargs.setdefault(key, value)

        if "local_prm_values" in grid_kwargs:
            local_prm_values = grid_kwargs["local_prm_values"]
            if "local_prm_value_weights" in grid_kwargs:
                local_prm_value_weights = grid_kwargs["local_prm_value_weights"]
            else:
                local_prm_value_weights = jnp.ones_like(local_prm_values)
        elif "local_value_min" in grid_kwargs and "n_local_values" in grid_kwargs:
            # make a grid of local parameter values, 
            # starting at lcoal_value_min and ending at local_value_max
            # equally spaced in cdf space
            xs_u = jnp.linspace(dist.Normal(0, 1).cdf(grid_kwargs['local_value_min']),
                                dist.Normal(0, 1).cdf(-1 * grid_kwargs['local_value_min']),
                                grid_kwargs['n_local_values'])
            local_prm_values = jnp.clip(dist.Normal(0, 1).icdf(xs_u), 
                                        min = grid_kwargs['local_value_min'],
                                        max = -1 * grid_kwargs['local_value_min'])
            local_prm_value_weights = jnp.ones_like(local_prm_values)
        else:
            local_prm_values, local_prm_value_weights = hermegauss(grid_kwargs.get("K", 32))

        full_data_model = partial(self.mcmc_model, control=self.default_control)

        # create function that runs posterior predictive checks for a single global sample
        pp_fun_for_sample = partial(_grid_postpred_for_sample,
                        local_prm_name = self.protect_model.local_prms[0],
                        local_prm_values = local_prm_values,
                        model = full_data_model,
                        data = self.data,
                        conditioning_sets = self.protect_model.conditioning_sets,
                        local_prm_value_weights = local_prm_value_weights,
                        deterministic_sites = deterministic_sites,
                        )

        # do vmap or scan
        if grid_kwargs['accelerator'] == "vmap":
            lls, site_values = vmap(pp_fun_for_sample)(global_samples)
        elif grid_kwargs['accelerator'] == "scan":
            def scannable_fn(carry, x):
                return carry, pp_fun_for_sample(x)
            _, (lls, site_values) = scan(scannable_fn, None, global_samples)

        return lls, site_values


    def get_c_index(self, pp_mode="no_y", posterior_samples=None, time=None, event=None):
        """
        calculate harell's c statistic for the model across all parameter values
        """
        if posterior_samples is None:
            if pp_mode not in self.mcmc_samples:
                print(f"No samples found for posterior predictive mode {pp_mode}. Running posterior_predictive with rng_key=PRNGKey(0).")
                rng_key = random.PRNGKey(0)
                posterior_samples = self.run_postpred_mcmc(rng_key, posterior_samples=self.mcmc_samples["posterior"], pp_mode=pp_mode)
            else:
                posterior_samples = self.mcmc_samples[pp_mode]
        

        if time is None:
            time = jnp.abs(self.data['time_cens'])
            event = self.data['time_cens'] > 0

        lps = posterior_samples["lp"]

        # if pp_mode=='no_txy', then lps has shape [2, num_samples, num_patients]
        # concatenate the first axis along the last axis, out_shape = [num_samples, 2*num_patients]
        if pp_mode == "no_txy":
            # TODO: when running on utrecht 607 pt data, this had shape (num_samples, 2, 1, 607)
            # but not for 504 data (which had (num_samples, 2, 504))
            # PATCH current patch: squeeze()
            lps = jnp.concatenate(jnp.transpose(lps.squeeze(), (1, 0, 2)), axis=-1)
            time = jnp.concatenate([time, time], axis=0)
            event = jnp.concatenate([event, event], axis=0)

        def _scan_fn(carry, x):
            return carry, harrell_c_streaming(time, x, event)
        _, c_index = scan(_scan_fn, None, lps)

        return c_index

    def get_treatment_auc(self, pp_mode="no_txy", posterior_samples=None, treatment=None):
        """
        calculate area under the curve (AUC) for the treatment predictions
        """
        if "tx" not in pp_mode:
            raise ValueError("AUC can only be calculated for posterior predictive modes that do not condition treatment, e.g. 'no_txy'")
        if posterior_samples is None:
            if pp_mode not in self.mcmc_samples:
                print(f"No samples found for posterior predictive mode {pp_mode}. Running posterior_predictive with rng_key=PRNGKey(0).")
                rng_key = random.PRNGKey(0)
                posterior_samples = self.run_postpred_mcmc(rng_key, posterior_samples=self.mcmc_samples["posterior"], pp_mode=pp_mode)
            else:
                posterior_samples = self.mcmc_samples[pp_mode]
        

        if treatment is None:
            treatment = self.data['tx']

        eta_tx = posterior_samples["eta_tx"]

        def _scan_fn(carry, x):
            return carry, roc_auc(treatment, x)
        _, auc = scan(_scan_fn, None, eta_tx)

        return auc







# helper functions for inference



def _grid_postpred_for_sample(
                        global_sample,
                        local_prm_name,
                        local_prm_values,
                        model,
                        data,
                        conditioning_sets,
                        local_prm_value_weights=None,
                        deterministic_sites = [],
                        model_kwargs = {}
                        ):
    """
    calculate likelihoods for posterior predictive modes for a single global sample
    local_prm_value_weights: weights for the local parameter values, if None, all values are equally weighted, e.g. use for gauss-hermite quadrature
    """

    # make the log likelihood function
    log_like_fn = _make_postpred_log_like_fn(model, data, local_prm_name, deterministic_sites, model_kwargs)
    # make log_joint function
    log_joint_fn = _make_log_joint_fn(conditioning_sets)
    # get weights
    weights = local_prm_value_weights if local_prm_value_weights is not None else jnp.ones_like(local_prm_values)

    # apply vectorized over all values for the local parameter, for a value of the global parameter
    ## log likelihoods of all observation sites and values of deterministic sites
    lls, deterministic_values = vmap(log_like_fn, in_axes=(0, None))(local_prm_values, global_sample)

    ## log joints for all conditioning sets
    log_joints = vmap(log_joint_fn)(lls)

    # calculate normalized joints
    log_joints_norm = {}
    log_weights = jnp.log(weights)
    for setname, log_joint in log_joints.items():
        # log_joint shape = (num_local_values, num_patients)
        lw = log_joint + log_weights[:, None]  # add weights to log joint
        # normalize the joint
        log_Z = logsumexp(lw, axis=0, keepdims=True)  # sum over local parameter values
        log_joint = lw - log_Z  # normalize the joint
        log_joints_norm[setname] = log_joint

    # calculate expected values for log likelihoods and deterministic sites per patient
    e_lls = {}
    e_sites = {}
    for setname, log_joint in log_joints_norm.items():
        # log_probs
        ell_set = {}
        for site_name, log_prob in lls.items():
            ell = logsumexp(log_prob + log_joint, axis=0)
            ell_set[site_name] = ell
        e_lls[setname] = ell_set

        # deterministic sites
        esites_set = {}
        for site_name, site_value in deterministic_values.items():
            esites_set[site_name] = jnp.average(site_value, weights=jnp.exp(log_joint), axis=0)
        e_sites[setname] = esites_set
            
    return e_lls, e_sites


def _make_postpred_log_like_fn(model, data, 
                               local_prm_name,
                               deterministic_sites=[], model_kwargs={}):
    """
    make a log likelihood function for posterior predictive modes
    the returned function takes as arguments a single value of the local parameter,
    and a sample of the global parameters
    and returns the log likelihoods of the model for a single value of the local_prm, the deterministic sites
    """

    # setup input data with / without treatment
    data0 = data | {'tx': jnp.zeros(data['tx'].shape)}
    data1 = data | {'tx': jnp.ones(data['tx'].shape)}

    def log_like_fn(local_prm_value, global_sample):
        # get model traces, both with and without enumeration (setting treatment to 0/1 for all observations)
        sub_model = handlers.substitute(model, data=global_sample | {local_prm_name: local_prm_value})

        # get traces
        tr = handlers.trace(sub_model).get_trace(data=data, **model_kwargs)
        tr0 = handlers.trace(sub_model).get_trace(data=data0, **model_kwargs)
        tr1 = handlers.trace(sub_model).get_trace(data=data1, **model_kwargs)

        # get likelihoods from observation sites
        lls = get_log_likelihoods_from_trace(tr)
        lls0 = get_log_likelihoods_from_trace(tr0)
        lls1 = get_log_likelihoods_from_trace(tr1)

        # get enumerated marginal survival likelihood by enumerating over treatment
        ll_y0_w = lls0['obs_y'] + lls0['obs_tx'] # log(p(y|tx=0)p(tx=0))
        ll_y1_w = lls1['obs_y'] + lls1['obs_tx'] # log(p(y|tx=1)p(tx=1))
        lls['y_enum'] = jnp.log(jnp.exp(ll_y0_w) + jnp.exp(ll_y1_w))
        # i.e. p(y) = sum_tau p(y|tau)p(tau)

        # get full conditional of site that is enumerated out
        # i.e. the full conditional of p(t|y,w,x) 
        # this is (expectation over F|y,w,x) of:  p(t,y|F,x) / (sum_tau p(y|tau,F,x)p(tau|F,x))
        # = (p(y|t,F,x)p(t|F,x)) / (sum_tau p(y|tau,F,x)p(tau|F,x))
        lls['tx_enum'] = lls['obs_y'] + lls['obs_tx'] - lls['y_enum']

        deterministic_values = {site: tr[site]["value"] for site in deterministic_sites}

        return lls, deterministic_values
    
    return log_like_fn

def _make_log_joint_fn(conditioning_sets):
    """
    make a log joint function based on conditioning sets; the returned function
    takes in a dictionary of log likelihoods for specific observation sites,
    and for each conditioning set (which is a subset of the observation sites),
    returns the log joint;
    also handled potential marginalization of a discrete variable t
    conditioning_sets: a dictionary with the conditioning sets for different postpred modes
    """
    # TODO PERF: this could potentially be made more efficient by replacing the for loop with e.g. a selector matrix
    def log_joint_fn(lls):
        # collect log_joints for different conditioning sets;
        # these correspond to different 'posterior' predictive modes, where we e.g. do not condition the latent factor on one of the proxies and/or treatment or outcome
        log_joints = {k: jnp.array(0.) for k in conditioning_sets.keys()}
        for site_name, log_prob in lls.items():
            # Replace NaNs with 0 so missing sites do not contribute to the joint
            log_prob = jnp.where(jnp.isnan(log_prob), 0.0, log_prob)
            for setname, conditioning_set in conditioning_sets.items():
                if site_name in conditioning_set:
                    log_joints[setname] += log_prob
        
        return log_joints

    return log_joint_fn

# define functions to be used in the inference class
# extract the first post_sample from the first chain

# setup set posterior predictive function with mcmc
def _make_scannable_mcmc_postpred_fn(rng_key, global_sample, pp_mode, model, data, control, obs_masks=None, mcmc_kwargs = {}):
    """
    make a jittable function for running postpred
    """
    # update defaults for postpred mcmc
    default_postpred_mcmc_kwargs = {
        "num_warmup": 100,
        "num_samples": 250,
        "num_chains": 1,
        "progress_bar": False,
        "jit_model_args": True
    }
    for key, value in default_postpred_mcmc_kwargs.items():
        mcmc_kwargs.setdefault(key, value)

    model_template = partial(model, data=data, obs_masks=obs_masks)
    pp_control = _get_ppcontrol(control, pp_mode)

    def pp_model(global_sample):
        # print(pp_control)
        with handlers.condition(data=global_sample):
            return model_template(control=pp_control)

    # run warmup (which will trigger jitting)
    pp_mcmc = MCMC(NUTS(pp_model), **mcmc_kwargs)
    pp_mcmc.warmup(rng_key, global_sample=global_sample)

    # run one posterior predictive modes on all data
    def scannable_ppfun(rng_key, global_sample):
        pp_key, carry_key = random.split(rng_key)
        local_sample = _get_local_sample_from_mcmcobj(
            pp_key, pp_mcmc, global_sample
        )
        return carry_key, local_sample

    return scannable_ppfun

def _make_mcmc_postpred_fn(global_sample, pp_mode, model, data, control, obs_masks=None, return_value='likelihoods', mcmc_kwargs = {}):
    """
    make a jittable function for running postpred
    return_value: 'likelihoods' or 'samples'
    """
    # update defaults for postpred mcmc
    default_postpred_mcmc_kwargs = {
        "num_warmup": 100,
        "num_samples": 250,
        "num_chains": 1,
        "progress_bar": False,
        "jit_model_args": True
    }
    for key, value in default_postpred_mcmc_kwargs.items():
        mcmc_kwargs.setdefault(key, value)

    model_template = partial(model, data=data, obs_masks=obs_masks)
    pp_control = _get_ppcontrol(control, pp_mode)

    def sample_fun(rng_key, post_samples):
        def pp_model(global_sample):
            with handlers.condition(data=global_sample):
                return model_template(control=pp_control)

        pp_mcmc = MCMC(NUTS(pp_model), **mcmc_kwargs)
        pp_mcmc.warmup(rng_key, global_sample=global_sample)

        # run one posterior predictive modes on all data
        def scannable_ppfun(rng_key, global_sample):
            pp_key, carry_key = random.split(rng_key)
            local_sample = _get_local_sample_from_mcmcobj(
                pp_key, pp_mcmc, global_sample
            )
            return carry_key, local_sample
        
        _, locals_samples = scan(scannable_ppfun, rng_key, post_samples)

        return locals_samples

    if return_value == 'samples':
        return sample_fun

    elif return_value == 'likelihoods':
        def pp_fun(rng_key, post_samples):
            locals_samples = sample_fun(rng_key, post_samples)

            lls = log_likelihood(partial(model_template, control=control), post_samples | locals_samples)
            lls_per_patient = _log_likelihood_per_patient(lls)

            return lls_per_patient
        return pp_fun

    elif return_value == 'both':
        # return both samples and lls per sample, usefull for bayesian updating
        def pp_fun(rng_key, post_samples):
            locals_samples = sample_fun(rng_key, post_samples)

            lls = log_likelihood(partial(model_template, control=control), post_samples | locals_samples)

            return locals_samples, lls
        return pp_fun
    else:
        raise ValueError("return_value must be 'samples', 'likelihoods' or 'both'")


def _make_baseline_fn(
    global_parameters,
    model,
    control,
    mcmc_kwargs = {}
                    ):
    """
    infer the baseline models for all observation nodes;
    these models only condition on direct parents of the observation nodes in the DAG, not on the latent factor
    used in Eqs 5-7 of the appendix of https://doi.org/10.1038/s41598-022-09775-9 
    :param global_parameters: global parameters
    :param F_parameters: parameters associated with the latent factor F
    :param mcmc_kwargs: kwargs for mcmc
    """

    # update mcmc_kwargs with default values only if they do not occur in mcmc_kwargs
    default_mcmc_kwargs = {"num_warmup": 250, "num_samples": 750, "num_chains": 1, "progress_bar": False} 
    for key, value in default_mcmc_kwargs.items():
        mcmc_kwargs.setdefault(key, value)

    # set the latent factor to zero
    # set the global parameters with F to zero
    F_params = {k: jnp.array(0.0) for k in global_parameters if k.startswith("b_F") or k.endswith("_F")}
    F_params["Feps"] = jnp.zeros(control["N"])
    flat_model = handlers.condition(partial(model, control=control), data=F_params)

    # run the model with the latent factor set to zero
    # get the samples
    mcmc = MCMC(NUTS(flat_model), **mcmc_kwargs)

    def baseline_fn(rng_key, data, obs_masks):
        mcmc.run(rng_key, data=data, obs_masks=obs_masks)
        samples = mcmc.get_samples(group_by_chain=False)

        # add in the samples that are all set to zero
        # first, for all F_params, broadcast them to the right shape, which is (num_chains * num_samples, )
        zero_samples = {k: jnp.zeros((mcmc_kwargs["num_chains"] * mcmc_kwargs["num_samples"],)) for k in F_params}

        return samples | zero_samples
    
    return baseline_fn

def _make_baseline_fn(
    global_parameters,
    model,
    control,
    mcmc_kwargs = {}
                    ):
    """
    infer the baseline models for all observation nodes;
    these models only condition on direct parents of the observation nodes in the DAG, not on the latent factor
    used in Eqs 5-7 of the appendix of https://doi.org/10.1038/s41598-022-09775-9 
    :param global_parameters: global parameters
    :param F_parameters: parameters associated with the latent factor F
    :param mcmc_kwargs: kwargs for mcmc
    """

    # update mcmc_kwargs with default values only if they do not occur in mcmc_kwargs
    default_mcmc_kwargs = {"num_warmup": 250, "num_samples": 750, "num_chains": 4, "progress_bar": False} 
    for key, value in default_mcmc_kwargs.items():
        mcmc_kwargs.setdefault(key, value)

    # set the latent factor to zero
    # set the global parameters with F to zero
    F_params = {k: jnp.array(0.0) for k in global_parameters if k.startswith("b_F") or k.endswith("_F")}
    F_params["Feps"] = jnp.zeros(control["N"])
    flat_model = handlers.condition(partial(model, control=control), data=F_params)

    # run the model with the latent factor set to zero
    # get the samples
    mcmc = MCMC(NUTS(flat_model), **mcmc_kwargs)

    def baseline_fn(rng_key, data, obs_masks):
        mcmc.run(rng_key, data=data, obs_masks=obs_masks)
        samples = mcmc.get_samples(group_by_chain=False)

        # add in the samples that are all set to zero
        # first, for all F_params, broadcast them to the right shape, which is (num_chains * num_samples, )
        zero_samples = {k: jnp.zeros((mcmc_kwargs["num_chains"] * mcmc_kwargs["num_samples"],)) for k in F_params}

        return samples | zero_samples
    
    return baseline_fn

def _get_local_sample_from_mcmcobj(
    rng_key, mcmc, global_sample, *args, **kwargs
):
    mcmc.run(rng_key, global_sample=global_sample, *args, **kwargs)
    ppsmps = mcmc.get_samples()
    ppsmps = {k: v[-1] for k, v in ppsmps.items()}  # grab only the last N samples
    return ppsmps


def _get_ppcontrol(inference_control, ppmode="no_y"):
    """
    helper function to generate a control dictionary for posterior predictions

    :param inference_control: control dictionary for inference
    :param ppmode: indicator for which variable to mask out
    :return: a new control dict
    """

    def toggle_sampling_statement(control, excludevar):
        """
        toggle sampling statement in the control dictionary
        """
        # TODO: instead of control arguments, use obs_masks to not sample a certain variable
        # then with same-shaped obs_masks, we can jit the model and get the same results
        # inference_control["enum_tx"] = True
        if excludevar == "tx":
            inference_control["sample_tx"] = False
            inference_control["enum_tx"] = True
        if excludevar == "txy":
            inference_control["sample_y"] = False
            inference_control["sample_tx"] = False
            inference_control["enum_tx"] = True
        else:
            inference_control["sample_" + excludevar] = False
        return control

    inference_control = inference_control.copy()
    if ppmode == "no_y":
        inference_control["sample_y"] = False
    elif ppmode == "full":
        # this is not technically a ppmode but re-running inference, can be used for external use of the model to refit locals (and optionally intercepts), using samples of the global parameters
        pass
    elif ppmode.startswith("no_"):
        # check if the ppmode has multiple instances of "no_" in it
        if ppmode.count("no_") == 1:
            excludevar = ppmode.split("no_")[-1]
            inference_control = toggle_sampling_statement(inference_control, excludevar)
        elif ppmode.count("no_") == 2:
            excludevars = ppmode.replace("no_", "").split("_")
            for var in excludevars:
                inference_control = toggle_sampling_statement(inference_control, var)
    else:
        raise NotImplementedError(f"found unrecognizable ppmode: {ppmode}")

    return inference_control



def _slice_posterior_for_pp(posterior_samples, global_sample_shape, num_samples_out=100, group_by_chain=False):
    """
    slice posterior samples for running posterior predictions where mcmc is required
    :param samples: dictionary with samples
    :param global_sample_shape: a tuple with the shape of the global samples (num_samples,) or (num_chains, num_samples) when group_by_chain=True
    :param num_samples_out: number of global parameters to get
    :param group_by_chain: group samples by chain?
    """
    # prepare the samples by slicing them
    if group_by_chain:
        num_chains = global_sample_shape[0]
        num_global_samples_per_chain = global_sample_shape[1]
    else:
        num_chains = 1
        num_global_samples_per_chain = global_sample_shape[0]

    num_global_samples = num_chains * num_global_samples_per_chain
    num_local_samples_per_chain = int(num_samples_out / num_chains)

    try:
        assert (num_samples_out <= num_global_samples)
    except AssertionError:
        raise ValueError(
            f"num_samples_out ({num_samples_out}) should be <= number of posterior samples ({num_global_samples}); global_sample_shape: {global_sample_shape}"
        )

    # create a slicing of the posterior samples to reduce auto-correlation in the samples in the new posterior draws
    ## starting with the last sample and going back
    stepsize = jnp.floor(num_global_samples_per_chain / num_local_samples_per_chain)
    revidxs = jnp.linspace(
        0,
        (num_local_samples_per_chain - 1) * stepsize,
        num=num_local_samples_per_chain,
    )
    sampleidxs = (num_global_samples_per_chain - revidxs - 1).astype(jnp.int32)
    takeaxis = 1 if group_by_chain else 0
    sliced_samples = {
        k: jnp.take(v, sampleidxs, axis=takeaxis)
        for k, v in posterior_samples.items()
    }

    return sliced_samples


def _log_likelihood_per_patient(lls, group_by_chain=False):
    # TODO: change this into util function in protect/utils.py
    """
    calculate log likelihoods per patient, from per-sample likelihoods
    """
    assert not group_by_chain, "group_by_chain=True not implemented yet"
    obs_sites = [k for k in lls.keys() if k.startswith("obs_")]
    num_samples = lls[obs_sites[0]].shape[0]

    lls_per_patient = {}
    # summarize across per site across observations
    lls_joint = jnp.zeros(lls[obs_sites[0]].shape)
    for obs_site, ll in lls.items():
        var_name = obs_site.replace("obs_", "")
        lls_per_observation = logsumexp(ll, axis=0) - jnp.log(num_samples)
        lls_joint += lls_per_observation
        lls_per_patient[var_name] = lls_per_observation

    # calculate joint likelihood over treatment and outcome
    lltxy = lls['obs_tx'] + lls['obs_y']
    lls_per_patient['txy'] = logsumexp(lltxy, axis=0) - jnp.log(num_samples)

    # calculate joint total likelihood
    lls_per_patient["joint"] = logsumexp(lls_joint, axis=0) - jnp.log(num_samples)

    return lls_per_patient



def _marginalize_hazard_ratio_pgw(rng_key, sample, maxtime=10, start=None):
    """
    given a posterior sample, calculate the marginalized hazard ratio
    # TODO: implement a non-pgw version of this function that uses the model to get the samples
    """

    alpha0 = sample["alpha0"]
    nu0 = sample["nu0"]
    beta0 = sample["beta0"]
    lp_notx = sample["lp_notx"]
    lp_dotx = sample["lp_dotx"]

    sample0 = {
        "beta": beta0 + lp_notx,
        "alpha0": alpha0,
        "nu0": nu0
    }
    sample1 = sample0 | {
        "beta": beta0 + lp_dotx
    }

    def sample_pgw(rng_key, params):
        """sample times from a power generalized weibull"""
        pgw = PGW(0.0, params['beta'], params["alpha0"], params["nu0"])
        return pgw.sample(rng_key, (1,)).squeeze()
    
    # sample times, and set maxtime
    rng0, rng1 = random.split(rng_key, 2)
    y0 = sample_pgw(rng0, sample0)
    y1 = sample_pgw(rng1, sample1)
    time_cens0 = time_event_to_time_cens(y0, maxtime=maxtime)
    time_cens1 = time_event_to_time_cens(y1, maxtime=maxtime)


    # determine start point for optimization
    if start is None:
        # use a small value for the start point
        # this is to avoid numerical issues with the optimization
        start = -1e-4 * jnp.ones(4)

    # calculate the hazard ratio by optimizing a pgw model
    # prepare fake RCT data by concatenating the two time_cens samples
    # stack time_cens samples 
    y = jnp.concatenate([time_cens0, time_cens1])
    tx = jnp.concatenate([jnp.zeros_like(time_cens0), jnp.ones_like(time_cens1)])
    Xmat = tx.reshape(-1,1)

    result = optimize_pgw(y, Xmat, start=start)

    return result, time_cens0, time_cens1
