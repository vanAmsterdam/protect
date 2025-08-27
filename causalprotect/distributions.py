'''
source code for likelihoods etc for manual models
'''

from jax import numpy as jnp, lax, random, custom_jvp

from numpyro.distributions import constraints, Normal, HalfNormal
from numpyro.distributions.transforms import AffineTransform
from numpyro.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions.util import (
    promote_shapes,
    validate_sample
)


def reshape_times_and_kappa(ti, kap):
    ## fix minimal number of dimensions
    if ti.ndim == 0:
        ti = ti.reshape((-1,))
    if kap.size == 1:
        kap = jnp.ones_like(ti) * kap
    return ti, kap

# define power-generalized weibull functions with analytic gradients (for numerical stability)
@custom_jvp
def pgwh0(ti,kap):
    '''
    baseline hazard
    
    :param ti: vector of times
    :param kap: kappa, single number or vector of length ti
    '''
    out = (1+ti/(kap+1))**(kap-1)
    ## fix infinite cases
    out = jnp.where(jnp.isinf(kap), jnp.exp(ti), out)
    
    return out

def pgwh0d(ti,kap):
    'time derivative of hazard function'
    out = pgwh0(ti,kap)*pgwm0d(ti,kap)
    return out

def pgwh0k(ti,kap):
    'kappa derivative of hazard function'
    out = pgwh0(ti,kap)*pgwm0k(ti,kap)
    return out

@custom_jvp
def pgwm0(ti,kap):
    'log hazard function'
    # ti, kap = reshape_times_and_kappa(ti, kap)
    out = (kap - 1)*jnp.log(1 + (ti / (kap + 1)))
    out = jnp.where(jnp.isinf(kap), ti, out)
    return out

def pgwm0d(ti,kap):
    'z-derivative of log-hazard function'
    out = (kap-1) / (kap + 1 + ti)
    out = jnp.where(jnp.isinf(kap), 1, out)
    return out

def pgwm0k(ti,kap):
    'kappa-derivative of log hazard function'
    out = pgwm0(ti,kap) / (kap - 1) - ti * pgwm0d(ti,kap) / (kap + 1)
    out = jnp.where(kap==0, 0, out)
    out = jnp.where(jnp.isinf(kap), 0, out)
    return out

@custom_jvp
def pgwH0(ti,kap):
    '''
    baseline cumulative hazard
    
    :param ti: vector of times
    :param kap: kappa, single number or vector of length ti
    '''
    
    k0 = kap == 0
    ## replace zeros with small values for numeric stability (fix this later)
    kap = jnp.where(k0, 1e-6, kap)
    out = ((kap+1)/kap)*((1+ti/(kap+1))**kap-1)
    ## replace zeros sites with correct calculation
    out = jnp.where(k0, jnp.log(1+ti), out)
    ## fix infinite cases
    out = jnp.where(jnp.isinf(kap), jnp.exp(ti) - 1, out)

    return out

def pgwH0k(ti,kap):
    'k-derivative of cumulative hazard function'
    # ti, kap = reshape_times_and_kappa(ti,kap)
    H0i = pgwH0(ti,kap)
    h0i = pgwh0(ti,kap)
    m0i = pgwm0(ti,kap)
    out = ((kap*H0i+kap+1)*m0i)/(kap*(kap-1)) - H0i/(kap*(kap+1)) - ti*h0i/(kap+1)
    out = jnp.where(kap==0, 0, out)
    out = jnp.where(jnp.isinf(kap), 0, out)
    return out

def pgwH0d(ti,kap):
    'ti-derivative of cumulative hazard function'
    # ti, kap = reshape_times_and_kappa(ti,kap)
    out = (ti / (kap + 1) + 1)**(kap - 1)
    out = jnp.where(jnp.isinf(kap), 0, out)
    return out

# now define the custom analysical derivatives
@pgwH0.defjvp
def pgwH0_jvp(primals, tangents):
    ti, kap = primals
    ti_dot, kap_dot = tangents
    primal_out  = pgwH0(ti, kap)
    tangent_out = pgwH0d(ti, kap) * ti_dot + pgwH0k(ti, kap) * kap_dot
    return primal_out, tangent_out

# analytical derivative of h0
@pgwh0.defjvp
def pgwh0_jvp(primals, tangents):
    ti, kap = primals
    ti_dot, kap_dot = tangents
    primal_out  = pgwh0(ti, kap)
    tangent_out = pgwh0d(ti, kap) * ti_dot + pgwh0k(ti, kap) * kap_dot
    return primal_out, tangent_out

# analytical derivative of m0
@pgwm0.defjvp
def pgwm0_jvp(primals, tangents):
    ti, kap = primals
    ti_dot, kap_dot = tangents
    primal_out  = pgwm0(ti, kap)
    tangent_out = pgwm0d(ti, kap) * ti_dot + pgwm0k(ti, kap) * kap_dot
    return primal_out, tangent_out

class PowerGeneralizedWeibull(Distribution):
    '''
    PowerGeneralizedWeibull distributions
    see https://arxiv.org/abs/1901.03212
    '''
    arg_constraints = {'phi': constraints.positive, 
                       'lam': constraints.positive, 
                       'gam': constraints.positive,
                       'kap': constraints.greater_than(-1.0)}
    support = constraints.real
    reparametrized_params = ['phi', 'lam', 'gam', 'kap']

    def __init__(self, phi=1., lam=1., gam=1., kap=-0.1, validate_args=None):
        '''
        note that all parameters are expected to be exponentiated if they stem from a (linear) regression model,
        moreover, kappa should be substracted by 1 to allow for negative values of kappa    

        :param phi: part depending on tau-regression (Accelerated Failure Time)
        :param lam: part depending on beta-regression (Proportional Hazards)
        :param gam: part depending on alpha-regression (shape parameter (allows for non-proportional hazards))
        :param kap: part depending on nu-regression    (shape parameter for baseline hazard)
        '''
        self.phi, self.lam, self.gam, self.kap = promote_shapes(phi, lam, gam, kap)
        if not isinstance(self.phi, jnp.ndarray):
            self.phi = jnp.array(self.phi, dtype=jnp.float32)
        if not isinstance(self.lam, jnp.ndarray):
            self.lam = jnp.array(self.lam, dtype=jnp.float32)
        if not isinstance(self.gam, jnp.ndarray):
            self.gam = jnp.array(self.gam, dtype=jnp.float32)
        if not isinstance(self.kap, jnp.ndarray):
            self.kap = jnp.array(self.kap, dtype=jnp.float32)
        
        batch_shape = lax.broadcast_shapes(jnp.shape(phi), jnp.shape(lam), jnp.shape(gam), jnp.shape(kap))
        super(PowerGeneralizedWeibull, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        u = random.uniform(key, shape=sample_shape + self.batch_shape)
        return (1/self.phi)*((self._pgwH0(-jnp.log(u)/self.lam, 1/self.kap))**(1/self.gam))

    @validate_sample
    def log_prob(self, value):
        # :param value: the observed values; currently to indicate censoring, 
        # values < 0 are considered censored (so values = [0.3, -0.2] <==> ({0.3, 0.2}, {1, 0}) in standard survival analysis notation)

        # find out which observations are censored, by convention: values < 0 are considered censored
        if not isinstance(value, jnp.ndarray):
            value = jnp.array(value)
        notcensored = value >= 0
        times       = jnp.abs(value)

        # calculate args
        zi  = (self.phi*times)**self.gam
        zi, kap = reshape_times_and_kappa(zi, self.kap)
        h0i = self._pgwh0(zi, kap)
        H0i = self._pgwH0(zi, kap)
        
        return notcensored*jnp.log((self.lam*self.gam*zi/times)*h0i) - self.lam*H0i

    @property
    def mean(self):
        return jnp.full(self.batch_shape, jnp.nan)

    @property
    def variance(self):
        return jnp.full(self.batch_shape, jnp.nan)
    
    def _pgwH0(self,ti,kap=None):
        '''
        cumulative hazard function
        '''
        kap = self.kap if kap is None else kap
        return pgwH0(ti,kap)

    def _pgwh0(self,ti,kap=None):
        '''
        hazard function
        '''
        kap = self.kap if kap is None else kap
        return pgwh0(ti,kap)

    def survival(self, times):
        '''
        cumulative survival function
        '''
        if not isinstance(times, jnp.DeviceArray):
            times = jnp.array(times)
        return jnp.exp(-self.lam*self._pgwH0((self.phi*times)**self.gam))

    def hazard(self, times):
        '''
        hazard function
        '''
        if not isinstance(times, jnp.DeviceArray):
            times = jnp.array(times)
        return self._pgwh0((self.phi*times)**self.gam)*self.lam*self.gam*self.phi*((self.phi*times)**(self.gam-1))

def PowerGeneralizedWeibullLog(tau=0.0, beta=0.0, alpha=0.0, nu=0.0, validate_args=None):
    '''
    the PowerGeneralizedWeibull distribution, parameterized with log parameters with unrestricted scale
    '''
    phi = jnp.exp(tau)
    lam = jnp.exp(beta)
    gam = jnp.exp(alpha)
    kap = jnp.exp(nu) - 1.0

    return PowerGeneralizedWeibull(phi, lam, gam, kap, validate_args=validate_args)


class NegativeHalfNormal(TransformedDistribution):
    arg_constraints = {'scale': constraints.positive}
    support = constraints.less_than_eq(0.0)
    reparametrized_params = ["scale"]

    def __init__(self, scale=1., validate_args=None):
        base_dist = HalfNormal(scale)
        self.scale = base_dist.scale
        super(NegativeHalfNormal, self).__init__(
            base_dist, AffineTransform(0, -1), validate_args=validate_args
        )

    @property
    def mean(self):
        return -self.base_dist.mean
    
    @property
    def variance(self):
        return self.base_dist.variance


class DummyNegativeHalfNormal(Distribution):
    '''
    this is just a HalfNormal distribution but then renamed
    '''
    reparametrized_params = ['scale']
    support = constraints.positive
    arg_constraints = {'scale': constraints.positive}

    def __init__(self, scale=1., validate_args=None):
        self._normal = Normal(0., scale)
        self.scale = scale
        super(DummyNegativeHalfNormal, self).__init__(batch_shape=jnp.shape(scale), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return jnp.abs(self._normal.sample(key, sample_shape))

    @validate_sample
    def log_prob(self, value):
        return self._normal.log_prob(value) + jnp.log(2)

    @property
    def mean(self):
        return jnp.sqrt(2 / jnp.pi) * self.scale

    @property
    def variance(self):
        return (1 - 2 / jnp.pi) * self.scale ** 2

        
