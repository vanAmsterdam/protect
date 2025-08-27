'''
define some utility functions
'''

from pathlib import Path
import numpyro
import jax
import numpyro.distributions as dist
from causalprotect.distributions import PowerGeneralizedWeibullLog as PGW, NegativeHalfNormal
import numpy as np
import pandas as pd
from jax import numpy as jnp, ops, random, lax
from math import inf
from functools import partial
from arviz import InferenceData
from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
import warnings
import xarray as xr


import funsor
from numpyro.distributions.util import is_identically_one


import yaml


funsor.set_backend("jax")

distribution_dict = {
    'Normal':      dist.Normal,
    'HalfNormal':  dist.HalfNormal,
    'HalfCauchy':  dist.HalfCauchy,
    'Exponential': dist.Exponential,
    'Beta':        dist.Beta,
    'LKJCholesky': dist.LKJCholesky,
    'MultivariateNormal': dist.MultivariateNormal,
    'Delta':              dist.Delta,
    'NegativeHalfNormal': NegativeHalfNormal,
    # 'NegativeHalfNormal': DummyNegativeHalfNormal,
    'Uniform':            dist.Uniform,
    # 'InducedDirichlet':   InducedDirichlet
}


def _check_binary(y_true):
    # Assumes labels are 0/1 (bool or int). Cast to float.
    y = jnp.asarray(y_true)
    return y.astype(jnp.float32)


def roc_auc(y_true, y_score, sample_weight=None):
    """
    JIT/vmap-safe, tie-aware ROC AUC.
    AUC = sum_{groups g} [ w_pos_g * (cum_neg_before_g) + 0.5 * w_pos_g * w_neg_g ] / (W_pos * W_neg)
    """
    y = _check_binary(y_true)
    s = jnp.asarray(y_score, dtype=jnp.float32)
    w = jnp.ones_like(s) if sample_weight is None else jnp.asarray(sample_weight, dtype=jnp.float32)

    # Sort by score ASCENDING (so "before" = strictly smaller score)
    order = jnp.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    w_sorted = w[order]

    pos_w = y_sorted * w_sorted
    neg_w = (1.0 - y_sorted) * w_sorted

    # Run-length encode equal scores: group ids 0..G-1
    # boundary_start[i]=True starts a new group at i
    boundary_start = jnp.concatenate([jnp.array([True]), s_sorted[1:] != s_sorted[:-1]])
    group_id = jnp.cumsum(boundary_start.astype(jnp.int32)) - 1  # 0-based
    n = s.shape[0]

    # Scatter-add per group into length-n buffers (extra tail is unused zeros)
    pos_w_group = jnp.zeros(n, dtype=jnp.float32).at[group_id].add(pos_w)
    neg_w_group = jnp.zeros(n, dtype=jnp.float32).at[group_id].add(neg_w)

    # Number of groups (dynamic); create a mask to zero-out unused tail
    num_groups = group_id[-1] + 1  # scalar int32
    valid_mask = (jnp.arange(n, dtype=jnp.int32) < num_groups).astype(jnp.float32)

    # Exclusive cumsum of negative weight by group => negatives with strictly smaller scores
    csum_neg = jnp.cumsum(neg_w_group)
    cum_neg_before = jnp.concatenate([jnp.array([0.0], dtype=jnp.float32), csum_neg[:-1]])

    # Contribution per group: wins vs smaller negatives + 0.5 * ties within group
    contrib = pos_w_group * (cum_neg_before) + 0.5 * pos_w_group * neg_w_group
    contrib = contrib * valid_mask  # keep only real groups

    # Normalize
    W_pos = jnp.sum(pos_w)
    W_neg = jnp.sum(neg_w)
    denom = W_pos * W_neg
    auc = jnp.where(denom > 0, jnp.sum(contrib) / denom, jnp.nan)
    return auc


def harrell_c(event_times, predicted_scores, event_observed=None):
    """
    Harrell's concordance index (C) in JAX (vectorized, O(n^2) memory/time).

    Parameters
    ----------
    event_times : array (n,)
        Observed times (event or censoring).
    predicted_scores : array (n,)
        Risk scores where higher = higher risk (earlier event expected).
        If your model outputs higher=better, pass -predictions.
    event_observed : array (n,), bool or {0,1}, optional
        1/True if event observed, 0/False if censored. If None, all events.

    Returns
    -------
    c_index : scalar float
        Concordance in [0,1]; NaN if no comparable pairs.
    """
    T = jnp.asarray(event_times, dtype=jnp.float32)
    S = jnp.asarray(predicted_scores, dtype=jnp.float32)
    if event_observed is None:
        E = jnp.ones_like(T, dtype=bool)
    else:
        E = jnp.asarray(event_observed, dtype=bool)

    # Basic shape check (eager in Python; fine under jit as static check)
    if T.shape != S.shape or T.shape != E.shape:
        raise ValueError("event_times, predicted_scores, and event_observed must have the same shape.")

    # Sort by time (ascending) so "later" means strictly greater in the upper triangle.
    order = jnp.argsort(T)
    T = T[order]
    S = S[order]
    E = E[order]

    # Pairwise matrices
    Ti = T[:, None]
    Tj = T[None, :]

    Si = S[:, None]
    Sj = S[None, :]

    Ei = E[:, None]  # whether row i is an observed event

    # Comparable pairs: i is an observed event and j has strictly later time
    comparable = Ei & (Tj > Ti)

    # Pairwise concordance score: 1 if Si>Sj, 0.5 if tie, 0 otherwise
    pair_score = jnp.where(Si > Sj, 1.0, jnp.where(Si == Sj, 0.5, 0.0))

    num_conc = jnp.sum(pair_score * comparable)
    num_comp = jnp.sum(comparable)

    return jnp.where(num_comp > 0, num_conc / num_comp, jnp.nan)


def harrell_c_streaming(event_times, predicted_scores, event_observed=None):
    """
    Same metric, but avoids building n×n matrices (better for large n).
    Still O(n^2) time, but O(n) memory; JIT-friendly via lax.fori_loop.
    """
    T = jnp.asarray(event_times, dtype=jnp.float32)
    S = jnp.asarray(predicted_scores, dtype=jnp.float32)
    if event_observed is None:
        E = jnp.ones_like(T, dtype=bool)
    else:
        E = jnp.asarray(event_observed, dtype=bool)

    if T.shape != S.shape or T.shape != E.shape:
        raise ValueError("event_times, predicted_scores, and event_observed must have the same shape.")

    order = jnp.argsort(T)
    T = T[order]
    S = S[order]
    E = E[order]

    n = T.shape[0]

    def body(i, acc):
        num_conc, num_comp = acc
        # Only rows where event is observed contribute as "earlier" member
        def when_event(_):
            # compare to strictly later times
            later = T > T[i]
            Sj = jnp.where(later, S, -jnp.inf)  # mask non-later with -inf
            # Count pairs only among 'later'
            comp_count = jnp.sum(later)

            # Concordance score against all j>i
            # (Si>Sj) counts 1, (Si==Sj) counts 0.5
            conc = jnp.sum((S[i] > Sj) * 1.0 * later) + 0.5 * jnp.sum((S[i] == Sj) * 1.0 * later)
            return (num_conc + conc, num_comp + comp_count)

        return lax.cond(E[i], when_event, lambda _: (num_conc, num_comp), operand=None)

    num_conc, num_comp = lax.fori_loop(0, n, body, (0.0, 0.0))
    return jnp.where(num_comp > 0, num_conc / num_comp, jnp.nan)

def model_check_evaluation_to_df(model_check_evaluation):
    """
    take a model_check_evaluation dictionary and convert it to a pandas DataFrame
    """
    df1 = pd.DataFrame(model_check_evaluation['set1']).T.rename_axis('obs_var')
    df2 = pd.DataFrame(model_check_evaluation['set2']).T.rename_axis('obs_var')
    return pd.concat({'set1': df1, 'set2': df2}, axis=0)


def array_to_df(arr, dim_names=[], coord_values=None):
    """
    Convert an n-dimensional numpy array to a pandas DataFrame.
    
    :param arr: numpy ndarray with 0-3 dimensions
    :param dim_names: list of dimension names (length must match arr.ndim)
    :param coord_values: optional dict mapping dim_name -> list of coordinate values
    :return: pandas DataFrame
    """
    # Handle 0D case (scalar)
    if arr.ndim == 0:
        return pd.DataFrame({'value': [arr.item()]})

    # Handle >0D case
    if coord_values is None:
        coord_values = {dim: np.arange(size) for dim, size in zip(dim_names, arr.shape)}
    
    # Create an xarray DataArray
    xarr = xr.DataArray(arr, dims=dim_names, coords=coord_values)
    
    # Convert to long-form DataFrame
    return xarr.to_dataframe(name='value').reset_index()


def get_det_sites_from_trace(model_trace, site_names=None):
    '''
    given the result of a model trace, return the deterministic sites
    '''
    if site_names is None:
        site_names = [v['name'] for v in model_trace.values()]

    det_sites = {}
    for site in model_trace.values():
        if site['name'] in site_names and site['type'] == 'deterministic':
            det_sites[site['name']] = site['value']
    return det_sites


def get_log_likelihoods_from_trace(model_trace):
    """
    given a model trace, get log likelihood from observation sites
    taken from numpyro source code and therefore not tested
    """
    return {
        name: site["fn"].log_prob(site["value"])
        for name, site in model_trace.items()
        if site["type"] == "sample" and site["is_observed"]
    }

def generate_cv_intrain_matrix(rng_key, num_obs, num_folds):
    """
    Generate cross-validation splits using jax.
    
    :param rng_key: jax.random.PRNGKey
    :param num_obs: int, total number of observations
    :param num_folds: int, number of folds
    :return: matrix of shape (num_folds, num_obs), each row contains a boolean mask for the observation being in the train fold
    """
    intrain_indices = generate_cv_splits(rng_key, num_obs, num_folds)
    in_train_list = []
    for i in range(num_folds):
        in_train = jnp.where(jnp.isin(jnp.arange(num_obs), intrain_indices[i]), True, False)
        in_train_list.append(in_train)
    return jnp.stack(in_train_list, axis=0)


def generate_cv_splits(rng_key, num_obs, num_folds):
    """
    Generate cross-validation splits using jax.
    
    :param rng_key: jax.random.PRNGKey
    :param num_obs: int, total number of observations
    :param num_folds: int, number of folds
    :return: list of arrays, each array contains indices for training samples in each fold
    """
    indices = jnp.arange(num_obs)
    shuffled_indices = random.permutation(rng_key, indices)
    # find the number of samples in each test fold (floor(num_obs/num_folds))
    test_fold_sizes = jnp.full(num_folds, num_obs // num_folds)
    
    test_folds = []
    start = 0
    for size in test_fold_sizes:
        test_folds.append(shuffled_indices[start:start + size])
        start += size
    
    train_indices = []
    for test_fold_indices in test_folds:
        train_indices.append(jnp.setdiff1d(indices, test_fold_indices))
    
    return train_indices


def summarize_likelihoods(lls_per_patient, obs_masks=None, in_test=None):
    """
    given likelihoods per patient, create a summary
    :param lls_per_patient: dictionary with likelihoods per patient for every observation site
    :param obs_masks: dictionary with observation masks, optional
    :param in_test: boolean array with test indices, optional
    """
    summary = {}

    for var_name, ll in lls_per_patient.items():
        is_obs = jnp.ones(ll.shape[0], dtype=bool)
        if obs_masks and var_name in obs_masks:
            is_obs *= obs_masks[var_name]
        if in_test is not None:
            in_train = ~in_test
        else:
            in_train = jnp.ones(ll.shape[0], dtype=bool)
        
        in_train_and_obs = in_train & is_obs
        # train_lls = ll[in_train & is_obs]
        summary[('train', var_name)] = {
            "sum": jnp.sum(ll, where=in_train_and_obs),
            "mean": jnp.mean(ll, where=in_train_and_obs),
            "std": jnp.std(ll, where=in_train_and_obs),
            "min": jnp.min(ll, where=in_train_and_obs, initial=1.0),
            "max": jnp.max(ll, where=in_train_and_obs, initial=-inf),
            "n_obs": (in_train & is_obs).sum(),
        }

        if in_test is not None:
            # test_lls = ll[in_test & is_obs]
            in_test_and_obs = in_test & is_obs
            summary[('test', var_name)] = {
                "sum": jnp.sum(ll, where=in_test_and_obs),
                "mean": jnp.mean(ll, where=in_test_and_obs),
                "std": jnp.std(ll, where=in_test_and_obs),
                "min": jnp.min(ll, where=in_test_and_obs, initial=1),
                "max": jnp.max(ll, where=in_test_and_obs, initial=-inf),
                "n_obs": (in_test & is_obs).sum(),
            }
    return summary


def time_event_to_time_cens(time, event=None, maxtime=None):
    """
    convert standard time, event notation for survival analysis to a single time_cens variable,
    where all non-events are coded as negative time; if maxtime is provided, censor times higher than maxtime
    """
    if event is None:
        event = jnp.ones_like(time)
    maxtime = jnp.nanmax(time) if maxtime is None else maxtime
    # first censor non-finite times
    event = jnp.where(~jnp.isfinite(time), 0, event)
    time = jnp.where(~jnp.isfinite(time), maxtime, time)
    # then censor times higher than maxtime
    event = jnp.where(time > maxtime, 0, event)
    time = jnp.clip(time, max=maxtime)
    time_cens = (2*event - 1) * time
    return time_cens


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def summarize_data(data):
    """
    summarize a dictionary of data arrays
    """
    shapes = {k: v.shape for k, v in data.items()}
    dims = {k: len(v.shape) for k, v in data.items()}
    means = {k: v.mean() for k, v in data.items()}
    stds = {k: v.std() for k, v in data.items()}
    nas = {k: np.isnan(v).sum() for k, v in data.items()}
    dtypes = {k: v.dtype for k, v in data.items()}
    summary = {'shapes': shapes, 'dims': dims, 'means': means, 'stds': stds, 'nas': nas, 'dtypes': dtypes}

    return summary

def check_data(data, metadata, obs_masks=None):
    """
    perform checks on the data against provided metadata
    """
    
    # check presence of two obligatory variables
    if 'tx' not in data:
        raise ValueError("variable 'tx' is not present in data, needs to be provided as a binary treatment indicator")
    if 'time_cens' not in data:
        raise ValueError("variable 'time_cens' is not present in data, needs to be provided. See the documentation on how to code this")

    data_summary = summarize_data(data)

    # dont allow missing data in control variables
    for k in metadata['control_variables']:
        if data_summary['nas'][k] > 0:
            if k == 'wtlossany' and metadata['model_control']['impute_wtlossany']:
                pass
            else:
                raise ValueError(f"missing data in control variable {k}")

    # check all variables are the same length
    if len(set(data_summary['shapes'].values())) > 1:
        unique_shapes = {}
        for k, v in data_summary['shapes'].items():
            if v not in unique_shapes.values():
                unique_shapes[k] = v
        raise ValueError(f"not all variables are the same length: shapes {unique_shapes}")

    # check all variables are 1d
    for k, v in data_summary['dims'].items():
        if v > 1:
            raise ValueError(f"variable {k} is {v}d")

    # check all proxies are boolean or 0/1, or nan
    for k in metadata['f_proxies']:
        isbinary = np.isin(data[k], [0, 1, np.nan])
        isnan = np.isnan(data[k])
        if not (isbinary | isnan).all():
            raise ValueError(f"variable {k} is not boolean or 0/1 or nan")

    # check treatment is boolean or 0/1
    if not np.isin(data['tx'], [0, 1]).all():
        raise ValueError("tx is not boolean or 0/1")

    # # check all variables are numeric or boolean
    for k, v in data.items():
        if (not np.issubdtype(v.dtype, np.number)) and (not np.isin(v, [0, 1]).all()):
            raise ValueError(f"variable {k} is not numeric or boolean")
       
    return True

def summary_likelihoods_to_df(likelihoods, dimnames=[]):
    """
    given a dictionary of likelihoods, return a pandas dataframe
    structure should be {ppmode: {("train/test", "obs_site"): {sum, mean, std, min, max, n_obs}}}
    if from multiple folds, the structure will be:
    {ppmode: {("train/test", obs_site): {"sum": [sum1, sum2, ...], "mean": [mean1, mean2, ...], ...}}}
    dimnames: optional tuple of string with names of dimensions of each array
    """

    lldfs = []
    for ppmode, pplls in likelihoods.items():
        lldfi = summary_likelihoods_to_df_per_ppmode(pplls, dimnames)
        lldfi['ppmode'] = ppmode
        lldfs.append(lldfi)

    lldf = pd.concat(lldfs, axis=0)

    return lldf


def summary_likelihoods_to_df_per_ppmode(likelihoods, dimnames=[]):
    """
    given a dictionary of likelihoods, return a pandas dataframe
    keys should be ("train/test", "obs_site")
    structure:

    {("train/test", "obs_site"): {"sum": ..., "mean": ... , std, min, max, n_obs}}
    dimnames: optional tuple of string with names of dimensions of each array

    """
    key0 = list(likelihoods.keys())[0]
    if len(key0) == 3:
        raise ValueError("this function is only for likelihoods with keys ('split', 'obs_site')")

    value_shape = next(iter(likelihoods[key0].values())).shape

    try:
        assert len(value_shape) == len(dimnames)
    except AssertionError:
        raise ValueError(
            f"length of dimnames {dimnames} does not match length of value_shape {value_shape}; please provide the correct number of dimension names, for example ['fold']"
        ) from AssertionError
    
    lldfs = []
    for split_obs, metrics in likelihoods.items():
        metric_dfs = []
        for metric_tag, metric_arr in metrics.items():
            if metric_arr.shape != value_shape:
                raise ValueError(f"shape of {metric_tag} {metric_arr.shape} does not match shape of {key0} {value_shape}")

            metric_dfi = array_to_df(metric_arr, dim_names=dimnames)
            metric_dfi['metric'] = metric_tag
            metric_dfs.append(metric_dfi)
        
        # concatenate all metric dataframes
        metric_df = pd.concat(metric_dfs, axis=0)
        metric_df['split'] = split_obs[0]
        metric_df['obs_site'] = split_obs[1]
        lldfs.append(metric_df)
    
    lldf_long = pd.concat(lldfs, axis=0)

    # convert to wide form by casting all metrics to columns
    lldf = lldf_long.pivot_table(index=['split', 'obs_site']+dimnames, columns='metric', values='value')

    return lldf.reset_index()


def get_log_joint_of_trace(model_trace, site_names=None):
    '''
    given the result of a model trace, return the log_joint of sites
    '''
    if site_names is None:
        site_names = [v['name'] for v in model_trace.values()]
    
    log_joint = jnp.array(0.)
    for site in model_trace.values():
        if site['name'] in site_names and site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            scale = site['scale']
            if intermediates:
                log_prob = site['fn'].log_prob(value, intermediates)
            else:
                log_prob = site['fn'].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            log_joint = log_joint + log_prob
    return log_joint


def get_log_probs_from_trace(model_trace, apply_scale=True):
    '''
    given the trace of a model (i.e. from trace(model).get_trace(...))
    retrieve log probs of sample sites
    :param model_trace: result of trace(model).get_trace(...)
    :param bool apply_scale: apply scaling if present in the trace
    '''
    log_probs = {}
    for site in model_trace.values():
        if site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            scale = site['scale']
            if intermediates:
                log_prob = site['fn'].log_prob(value, intermediates)
            else:
                log_prob = site['fn'].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)) and apply_scale:
                log_prob = scale * log_prob

            log_probs[site['name']] = log_prob
    return log_probs


def pgw_criterion(prms, y, Xmat, mask, lambd=0.0):
    '''
    given parameter values and data, setup a criterion based on the powergeneralized weibull model for optimization
    :param jnp.ndarray prms: jnp.array of parameter values
    :param jnp.ndarray y: 1-D survival vector (ifelse(event==1, time, -time))
    :param jnp.ndarray Xmat:  2-D predictor matrix
    :param jnp.ndarray mask: 1-D bool vector of size y.shape[0] to indicate which observations to keep (1 = keep)
    :param float lambd: optional lambda for l2 regularization on betas
    '''
    beta0 = prms[0]
    alpha0, nu0 = prms[-2:]
    betas = prms[1:-2].reshape(-1,1)

    lps = jnp.dot(Xmat, betas).squeeze()
    lps += beta0

    logprob = jnp.where(mask, PGW(0.0, lps, alpha0, nu0).log_prob(y), 0)
    logprob_mean = logprob.sum() / mask.sum()
    penalty = lambd * (betas ** 2).sum()
    return penalty - logprob_mean


def optimize_pgw(y, Xmat, lambd=0.0, start=None, mask=None):
    '''
    optimize the PowerGeneralized Weibull model with some matrix of predictors
    :param jnp.ndarray y: 1-D survival vector (ifelse(event==1, time, -time))
    '''

    if mask is None:
        mask = jnp.ones_like(y, dtype=bool)
    if start is None:
        # cannot start the pgw-model at 0 for nu0 as the gradient is undefined there
        start = -1e-4 * jnp.ones(Xmat.shape[1] + 3) # (dim(X) + 3 intercepterms)

    criterion_fn = partial(pgw_criterion, y=y, Xmat=Xmat, mask=mask, lambd=lambd)
    # crit2 = lambda x, y: criterion_fn(x)
    def optim_fun(prms):
        return jax.value_and_grad(criterion_fn)(prms)

    results = tfp.optimizer.bfgs_minimize(optim_fun, initial_position=start, tolerance=1e-5, max_iterations=100)
    # results = jax.scipy.optimize.minimize(criterion_fn, x0=start, method="BFGS") 

    return results




def samples_to_df(x):
    '''
    given an array of samples of shape (nchain, ndraw, nobs), return a melted data.frame with one row per chain, draw and obs
    '''
    if x.ndim == 3:
        nchain, ndraw, nobs = x.shape
        df = pd.concat(
                dict(zip(range(nchain),
                    # per chain
                    [pd.concat(dict(zip(range(ndraw), [pd.DataFrame({'value': x[chaini,drawi]}) for drawi in range(ndraw)]))) for chaini in range(nchain)])))

        df = df.reset_index().rename({
            'level_0': 'chain',
            'level_1': 'draw',
            'level_2': 'obs'},
            axis=1)
    elif x.ndim == 2:
        nchain, ndraw = x.shape
        df = pd.concat(
                dict(zip(range(nchain),
                    # per chain
                    [pd.DataFrame({'value': x[chaini]}) for chaini in range(nchain)])))

        df = df.reset_index().rename({
            'level_0': 'chain',
            'level_1': 'draw'},
            axis=1)
    else:
        raise NotImplementedError("only works for samples shaped (nchain, ndraw, nobs) or (nchain, ndraw, nobs) or (nchain, ndraw)")
    return df


def sample_dict_to_df(d, verbose=False):
    '''
    given a dictionary of named samples, return a single melted data.frame with one row per chain, draw (obs) and parameter
    '''
    dfs = {}
    if verbose:
        for k, v in tqdm(d.items()):
            print(k)
            dfs[k] = samples_to_df(v)
    else:
        for k, v in d.items():
            dfs[k] = samples_to_df(v)

    df = pd.concat(dfs).reset_index().drop('level_1', axis=1).rename({'level_0': 'parameter'}, axis=1)
    return df


def time_and_event_to_binary(time, event, cutoff):
    '''
    given standard survival time notation (time, event),
    return binary outcome at some cutoff
    '''
    assert len(jnp.unique(event)) < 3, f"must be binary outcome; unique outcomes: {jnp.unique(event)}"

    # create empty array with nans
    yout = jnp.empty_like(time)
    yout = ops.index_update(yout, ops.index[:], jnp.nan)

    # follow-up greater than cutoff
    yout = ops.index_update(yout, jnp.where(time > cutoff)[0], 0.0)
    # events before cutoff
    yout = ops.index_update(yout, jnp.where((time < cutoff) & (event == 1))[0], 1.0)

    return yout



def add_txy_to_azd(azd, safemode=False):
    '''
    given an arviz InferenceData object with log likelihoods for observation sites obs_y and obs_tx,
    return the object with the joint of obs_y and obs_tx together
    :param azd: arviz.InferenceData object
    :param safemode: raise error if no obs_y and obs_tx in azd (if false, this returns the original azd object)
    :returns: arviz.InferenceData object
    '''
    assert isinstance(azd, InferenceData)
    try:
        lly = azd.log_likelihood.obs_y.values
        lltx = azd.log_likelihood.obs_tx.values
        lltxy = lly + lltx
        dimnames = ('chain', 'draw', 'txy_dim_0')
        azd.log_likelihood.__setitem__('txy', (dimnames, lltxy))
    except Exception as e:
        if safemode:
            raise e
        else:
            pass
    
    return azd




def get_global_parnames(x):
    if isinstance(x, dict):
        return get_global_parnames_dict_(x)
    elif isinstance(x, InferenceData):
        return get_global_parnames_azd_(x)
    else:
        raise NotImplementedError(f"dont know how to extract global parameter names from type {type(x)}")

def get_global_parnames_dict_(smpdict=None):
    '''
    given an arviz dataset with a posterior or a dictionary of samples, get all global parameter names
    this relies on the heuristic that there should be a parameter named 'lp'
    '''
    allvarnames = list(smpdict.keys())
    if 'lp' not in allvarnames:
        key0 = allvarnames[0]
        num_chains, num_samples = smpdict[key0].shape[:2]
        num_obs = 5
        warnings.warn('guessing the max dimension of a parameter is 5 as there is no lp in the posterior')
    else:
        num_chains, num_samples, num_obs = smpdict['lp'].shape
    globalpars = []
    globalshapes = []
    for varname, smpi in smpdict.items():
        isglobal = False
        varshape = smpi.shape
        if varshape == (num_chains, num_samples):
            isglobal = True
        elif len(varshape) == 3:
            if num_obs is not None:
                if varshape[2] < num_obs:
                    # this is to allow for multidemensional parameters
                    isglobal = True
            else:
                raise ValueError(f"cannot discern globals from locals when no lp in allvarnames {allvarnames}")
        if isglobal:
            globalpars.append(varname)
            globalshapes.append(varshape)

    # print(dict(zip(globalpars, globalshapes))
    return globalpars

def get_global_parnames_azd_(azd):
    '''
    given an arviz dataset with a posterior, get all global parameter names
    this relies on the heuristic that there should be a parameter named 'lp'
    '''
    allvarnames = list(azd.posterior.keys())
    if 'lp' not in allvarnames:
        key0 = allvarnames[0]
        num_chains, num_samples = azd.posterior.get(key0).shape[:2]
        num_obs = 5
        warnings.warn('guessing the max dimension of a parameter is 5 as there is no lp in the posterior')
    else:
        num_chains, num_samples, num_obs = azd.posterior.lp.shape
    globalpars = []
    globalshapes = []
    for varname in allvarnames:
        isglobal = False
        varshape = azd.posterior.get(varname).shape
        if varshape == (num_chains, num_samples):
            isglobal=True
        elif len(varshape) == 3:
            if num_obs is not None:
                if varshape[2] < num_obs:
                    # this is to allow for multidemensional parameters
                    isglobal = True
            else:
                raise ValueError(f"cannot discern globals from locals when no lp in allvarnames {allvarnames}")
        if isglobal:
            # print(f"found global variable {varname} with shape {varshape}")
            globalpars.append(varname)
            globalshapes.append(varshape)

    # print(dict(zip(globalpars, globalshapes))
    return globalpars

def expand_if_nd(x, n=1):
    if x.ndim > n:
        out = x
    else:
        add_dims = tuple([d for d in range(n+1) if d >= x.ndim])
        out = jnp.expand_dims(x, add_dims)
    return out

def expand_if_1d(x):
    return expand_if_nd(x, n=1)

def make_dummy_priorfunc(prm_names, dist=dist.Normal(0,5), verbose=False):
    '''
    generate a dummy prior function
    '''
    def prior_fn():
        prms = {}
        for k in prm_names:
            if verbose:
                print(f"prm: {k}", end='')
            prms[k] = numpyro.sample(k, dist)
            if verbose:
                print(f" val: {prms[k]}")
        return prms
    return prior_fn


def azd_to_dict(azd, prm_names=None, group='posterior', to_jnp=True, verbose=False):
    '''
    take arviz data loaded from disk, convert to dictionary

    :param azd: arviz data
    :param prm_names: list of parameters to take
    :param group: data group in arviz (posterior, posterior_predictive or constant_data)
    :param to_jnp: convert arrays to jax numpy arrays
    :param verbose: print progress etc
    '''
    out = {}
    if prm_names is None:
        if group == 'posterior':
            allvars = azd.posterior._variables.keys()
        elif group=='posterior_predictive':
            allvars = azd.posterior_predictive._variables.keys()
        elif group=='constant_data':
            allvars = azd.constant_data._variables.keys()
        elif group=='log_likelihood':
            allvars = azd.log_likelihood._variables.keys()
        else:
            raise NotImplementedError
        prm_names = [x for x in allvars if not ('_dim_' in x or x in ['chain', 'draw'])]

    if verbose:
        for prm_name in tqdm(prm_names):
            try:
                if group == 'posterior':
                    val = azd.posterior.get(prm_name).values
                elif group == 'posterior_predictive':
                    val = azd.posterior_predictive.get(prm_name).values
                elif group == 'constant_data':
                    val = azd.constant_data.get(prm_name).values
                elif group == 'log_likelihood':
                    val = azd.log_likelihood.get(prm_name).values
                else:
                    raise NotImplementedError(f"choose group from [posterior, posterior_predictive], you picked {group}")
                # if val.ndim < 2 and val.size == 1:
                    # val = val.reshape(1,1)
                if to_jnp:
                    val = jnp.array(val)
                out[prm_name] = val
                print(f"{prm_name}: {val.shape}")
            except Exception as e:
                print(f"error while fetching {prm_name}")
                print(f"allvars: {allvars}")
                raise e
    else:
        for prm_name in prm_names:
            try:
                if group == 'posterior':
                    val = azd.posterior.get(prm_name).values
                elif group == 'posterior_predictive':
                    val = azd.posterior_predictive.get(prm_name).values
                elif group == 'constant_data':
                    val = azd.constant_data.get(prm_name).values
                elif group == 'log_likelihood':
                    val = azd.log_likelihood.get(prm_name).values
                else:
                    raise NotImplementedError(f"choose group from [posterior, posterior_predictive], you picked {group}")
                # if val.ndim < 2 and val.size == 1:
                    # val = val.reshape(1,1)
                if to_jnp:
                    val = jnp.array(val)
                out[prm_name] = val
                # print(f"{prm_name}: {val.shape}")
            except Exception as e:
                print(f"error while fetching {prm_name}")
                print(f"allvars: {allvars}")
                raise e
    return out


def print_trace_obj(tr, N=100):
    '''
    helper function for printing results of a trace
    N: number of observations for determining how to print stuff
    '''
    for k, v in tr.items():
        if v['type'] == 'sample':
            print(f"\nsite: {k:10}", end="")
            print(f" is observed: {v['is_observed']:1}", end="")
            print(f", shape: {str(v['value'].shape):10}", end="")
            if not v['is_observed']:
                if v['value'].ndim > 0:
                    if v['value'].shape[0] < N:
                        print(f" value: {v['value']}", end="")
    print('')




def unpack_parameter_names(prm_names: list):
    '''
    given a list of parameter names, unpack some details about the parameter
    e.g. b_F_y    ->  (type, source, target) = (b, F, y)
    e.g. b_Ftx_y ->   (type, source, target) = (b, Ftx, y) 
    e.g. mu_y ->      (type, source, target) = (mu, NaN, y) 
    '''

    # define type based on first part (b, mu or s)
    prmspec = pd.DataFrame(pd.Series([x.split('_')[0] for x in prm_names], index=prm_names, name='prm_type'))
    
    # infer source as second part
    prmspec.loc[prmspec.index.str.find('_')>0, 'prm_source'] = prmspec.loc[prmspec.index.str.find('_')>0].index.map(lambda x: x.split('_')[1])

    # target is third part
    prmspec['prm_target'] = prmspec.index.map(lambda x: x.split('_')[-1])
    prmspec.loc[~prmspec.prm_type.isin(['b']), 'prm_source'] = jnp.nan

    # get the shapes


    return prmspec

def update_priorspec(priorspec, data_shapes=None):
    '''
    given a pandas data.frame that contains the definition of priors, add some columns with extra info
    
    :param priorspec pandas.core.DataFrame: a data.frame containing prior defitions; index should be prm_name, columns: prior_dist, prior_prm1, prior_prm2
    :param data_shapes dict: an optional dict of data shapes
    :returns: a pandas.core.DataFrame with extra columns
    '''
    prmspec   = unpack_parameter_names(priorspec.index.to_list())
    priorspec = priorspec.join(prmspec)

    # create a tuple of params  of varying length
    priorspec['prm_tuple'] = priorspec[['prior_prm1', 'prior_prm2']].apply(lambda x: (*x.dropna(),), axis=1)

    if data_shapes:
        for k, v in data_shapes.items():
            if len(v) == 1:
                data_shapes[k] = (v[0],) + (1,)
            elif len(v) > 0:
                raise ValueError(f"currently only 1 dimensional priors are implemented, found {len(v)} dimensions for variable {k}")
        data_dims   = {k: v[1] for k, v in data_shapes.items()}
        priorspec['target_dim'] = priorspec.prm_target.map(data_dims)
        priorspec.loc[~priorspec.prm_source.isna(), 'source_dim'] = priorspec.loc[~priorspec.prm_source.isna(), 'prm_source'].map(data_dims)
        priorspec['prm_shape'] = priorspec[['source_dim', 'target_dim']].apply(lambda x: (*x.dropna().astype(int),), axis=1)

    return priorspec

def get_priors(priorspec):
    '''
    given a pandas data.frame that contains the definition of priors, get the priors
    
    :param priorspec pandas.core.DataFrame: a data.frame containing prior defitions; index should be prm_name, columns: prior_dist, prior_prm1, prior_prm2
    :returns: a dictionary of numpyro.dist objects
    '''
    priors = {}
    # iterate over parameters
    for prm_name, prm_info in priorspec.iterrows():
        # grab class definition from dict
        prior_dist = distribution_dict[prm_info['prior_dist']]
        
        # instantiate class with defined parameters
        # print(f"getting prior for {prm_name} with info {prm_info['prm_tuple']}")
        try:
            prior = prior_dist(*prm_info['prm_tuple'])
        except Exception as e:
            print(f"error in prior for {prm_name}, with parameter info {prm_info}")
            raise e
        priors[prm_name] = prior
    
    return priors

# create a function that returns the parameters from priors in a dictionary
def create_prior_func(priors):
    '''
    create a separate function for sampling priors
    priors: a dict with priors (numpyro.distribution.Distribution)
    '''
    def prior_func():
        '''
        a funcion that returns samples from priors
        '''
        prms = {prm: numpyro.sample(prm, prior) for prm, prior in priors.items()}

        return prms
    return prior_func


def priorspec_to_priorfunc(priorspec: Path|str):
    '''
    given a path to a csv file with prior definitions, return a function that samples from these priors
    '''
    priorspec = pd.read_csv(priorspec, index_col='prm_name')
    priorspec = update_priorspec(priorspec)
    priors = get_priors(priorspec)
    prior_func = create_prior_func(priors)

    return prior_func

