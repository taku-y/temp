# -*- coding: utf-8 -*-

"""PyMC3 probability model of mixture of Bayesian mixed LiNGAM models.

In short, the model is referred to as mixutre BML models.
"""
# Author: Taku Yoshioka
# License: MIT

import numpy as np
from pymc3 import Normal, StudentT, HalfNormal, DensityDist, \
                  Deterministic, Gamma, Lognormal, Uniform, Beta
from pymc3.math import logsumexp
from scipy.special import gamma, gammaln
import theano
import theano.tensor as tt
from theano.tensor import sgn
from theano.tensor.slinalg import cholesky

from bmlingam.utils import standardize_samples

floatX = theano.config.floatX


class MixBMLParams:
    u"""Hyperparameters of BML mixture model.

    :param str dist_indvdl: Distribution of individual specific effects,
        :code:`gauss`, :code:`t` or :code:`gg`. Default is :code:`'t'`.
    :param float df_indvdl: Degrees of freedom of T-distribution on individual
        specific effects. This is used when dist_indvdl is :code:`t`.
        Default is 8.0.
    :param str dist_beta_indvdl: Distribution of the shape parameter of the
        generalized Gaussian distribution on individual specific effects.
        This parameter is used when dist_indvdl is :code:`gg`.
        Default is :code:`'uniform, 0.1, 1.0'`.
    :param str dist_l_cov_21: Distribution of correlation coefficients of
        individual specific effects. Default is :code:`'uniform, -0.9, 0.9'`.
    :param str dist_scale_indvdl: Distribution of the scale of individual
        specific effects. Default is :code:`'uniform, 0.01, 1.0'`.
    :param str dist_noise: Distribution of observation noise,
        :code:`laplace` or :code:`gg`. Default is :code:`laplace`.
    :param str dist_beta_noise: Distribution of the shape parameter of the
        generalized Gaussian distribution on observation noise.
        This parameter is used when dist_noise is :code:`gg`.
    :param str dist_std_noise: Distribution of the standard deviation of
        observation noise. The possible values are :code:`tr_normal`,
        :code:`log_normal` and :code:`uniform`. Default is :code:`log_normal`.

    :param bool standardize: If :code:`True`, observed samples are
        standardize before inference. Thus samples of RVs from variational
        posterior should be appropriately scaled. Default is :code:`True`.
    :param bool subtract_mu_reg: If :code:`True`, in regression, the common
        interception and the means of individual specific effects are subtracted
        from independent variables. Default is :code:`False`.
    :param bool fix_mu_zero: If :code:`True`, the common interception is fixed
        to 0. Default is :code:`True`.
    :param str prior_var_mu: How to set the prior variance of common
        interceptions. Default is :code:`'auto'`.
    """
    def __init__(self, dist_indvdl='t', df_indvdl=8.0,
        dist_beta_indvdl='uniform, 0.1, 1.0',
        dist_l_cov_21='uniform, -0.9, 0.9',
        dist_scale_indvdl='uniform, 0.01, 1.0',
        dist_noise='laplace', dist_beta_noise='uniform, 0.01, 1.0',
        dist_std_noise='log_normal', standardize=True, subtract_mu_reg=False,
        fix_mu_zero=True, prior_var_mu='auto'):
        self.dist_indvdl = dist_indvdl
        self.df_indvdl = df_indvdl
        self.dist_beta_indvdl = dist_beta_indvdl
        self.dist_l_cov_21 = dist_l_cov_21
        self.dist_scale_indvdl = dist_scale_indvdl
        self.dist_noise = dist_noise
        self.dist_beta_noise = dist_beta_noise
        self.dist_std_noise = dist_std_noise
        self.standardize = standardize
        self.subtract_mu_reg = subtract_mu_reg
        self.fix_mu_zero = fix_mu_zero
        self.prior_var_mu = prior_var_mu

    def as_dict(self):
        d = dict()
        d.update({'dist_indvdl': self.dist_indvdl})
        d.update({'df_indvdl': self.df_indvdl})
        d.update({'dist_beta_indvdl': self.dist_beta_indvdl})
        d.update({'dist_l_cov_21': self.dist_l_cov_21})
        d.update({'dist_scale_indvdl': self.dist_scale_indvdl})
        d.update({'dist_noise': self.dist_noise})
        d.update({'dist_beta_noise': self.dist_beta_noise})
        d.update({'dist_std_noise': self.dist_std_noise})
        d.update({'standardize': self.standardize})
        d.update({'subtract_mu_reg': self.subtract_mu_reg})
        d.update({'fix_mu_zero': self.fix_mu_zero})
        d.update({'prior_var_mu': self.prior_var_mu})

        return d


def _dist_from_str(name, dist_params_):
    if type(dist_params_) is str:
        dist_params = dist_params_.split(',')

        if dist_params[0].strip(' ') == 'uniform':
            rv = Uniform(name, lower=float(dist_params[1]),
                               upper=float(dist_params[2]))
        else:
            raise ValueError("Invalid value of dist_params: %s" % dist_params_)

    elif type(dist_params_) is float:
        rv = dist_params_

    else:
        raise ValueError("Invalid value of dist_params: %s" % dist_params_)

    return rv


def _laplace_loglike(mu, b):
    u"""Returns 1-dimensional likelihood function of Laplace distribution.
    """
    def likelihood(xs):
        return tt.sum(-tt.log(2 * b) - abs(xs - mu) / b)

    return likelihood


def _gg_loglike(mu, beta, std):
    u"""Returns 1-dimensional likelihood function of generalized Gaussian.

    :param mu: Mean.
    :param beta: Shape parameter.
    :param std: Standard deviation.
    """
    def likelihood(xs):
        m = std**2 * gamma(0.5 / beta) / \
            ((2**(1 / beta)) * gamma(3 / (2 * beta)))
        normalize = - np.log(beta) - gammaln(0.5) \
                  + np.pi**(0.5) + gammaln(0.5 / beta) \
                  + (0.5 / beta) * np.log(2) + 0.5 * np.log(m)

        return - 0.5 * tt.power((xs**2) / m, beta) - normalize

    return likelihood


def _indvdl_t(hparams, std_x, n_samples, L_cov, verbose=0):
    df_L = hparams.df_indvdl
    dist_scale_indvdl = hparams.dist_scale_indvdl    
    scale1 = std_x[0] * _dist_from_str('scale_mu1s', dist_scale_indvdl)
    scale2 = std_x[1] * _dist_from_str('scale_mu2s', dist_scale_indvdl)

    scale1 = scale1 / np.sqrt(df_L / (df_L - 2))
    scale2 = scale2 / np.sqrt(df_L / (df_L - 2))

    u1s = StudentT('u1s', nu=np.float32(df_L), shape=(n_samples,), 
                   dtype=floatX)
    u2s = StudentT('u2s', nu=np.float32(df_L), shape=(n_samples,), 
                   dtype=floatX)

    L_cov_ = cholesky(L_cov).astype(floatX)
    tt.set_subtensor(L_cov_[0, :], L_cov_[0, :] * scale1, inplace=True)
    tt.set_subtensor(L_cov_[1, :], L_cov_[1, :] * scale2, inplace=True)
    mu1s_ = Deterministic('mu1s_', 
                          L_cov_[0, 0] * u1s + L_cov_[0, 1] * u2s)
    mu2s_ = Deterministic('mu2s_', 
                          L_cov_[1, 0] * u1s + L_cov_[1, 1] * u2s)

    if 10 <= verbose:
        print('StudentT for individual effect')
        print('u1s.dtype = {}'.format(u1s.dtype))
        print('u2s.dtype = {}'.format(u2s.dtype))

    return mu1s_, mu2s_


def _indvdl_gauss(hparams, std_x, n_samples, L_cov, verbose=0):
    dist_scale_indvdl = hparams.dist_scale_indvdl
    scale1 = std_x[0] * _dist_from_str('scale_mu1s', dist_scale_indvdl)
    scale2 = std_x[1] * _dist_from_str('scale_mu2s', dist_scale_indvdl)

    u1s = Normal(
        'u1s', mu=np.float32(0.), tau=np.float32(1.), 
        shape=(n_samples,), dtype=floatX
    )
    u2s = Normal(
        'u2s', mu=np.float32(0.), tau=np.float32(1.), 
        shape=(n_samples,), dtype=floatX
    )
    L_cov_ = cholesky(L_cov).astype(floatX)
    tt.set_subtensor(L_cov_[0, :], L_cov_[0, :] * scale1, inplace=True)
    tt.set_subtensor(L_cov_[1, :], L_cov_[1, :] * scale2, inplace=True)
    mu1s_ = Deterministic('mu1s_', 
                          L_cov[0, 0] * u1s + L_cov[0, 1] * u2s)
    mu2s_ = Deterministic('mu2s_', 
                          L_cov[1, 0] * u1s + L_cov[1, 1] * u2s)

    if 10 <= verbose:
        print('Normal for individual effect')
        print('u1s.dtype = {}'.format(u1s.dtype))
        print('u2s.dtype = {}'.format(u2s.dtype))

    return mu1s_, mu2s_


def _indvdl_gg(hparams, std_x, n_samples, L_cov, verbose):
    dist_scale_indvdl = hparams.dist_scale_indvdl
    scale1 = std_x[0] * _dist_from_str('scale_mu1s', dist_scale_indvdl)
    scale2 = std_x[1] * _dist_from_str('scale_mu2s', dist_scale_indvdl)

    # Uniform distribution on sphere
    gs = Normal('gs', np.float32(0.0), np.float32(1.0), 
                shape=(n_samples, 2), dtype=floatX)
    ss = Deterministic('ss', gs + sgn(sgn(gs) + np.float32(1e-10)) * 
                             np.float32(1e-10))
    ns = Deterministic('ns', ss.norm(L=2, axis=1)[:, np.newaxis])
    us = Deterministic('us', ss / ns)

    # Scaling s.t. variance to 1
    n = 2 # dimension
    beta = np.float32(hparams['beta_coeff'])
    m = n * gamma(0.5 * n / beta) \
        / (2 ** (1 / beta) * gamma((n + 2) / (2 * beta)))
    L_cov_ = (np.sqrt(m) * cholesky(L_cov)).astype(floatX)

    # Scaling to v_indvdls
    tt.set_subtensor(L_cov_[0, :], L_cov_[0, :] * scale1, inplace=True)
    tt.set_subtensor(L_cov_[1, :], L_cov_[1, :] * scale2, inplace=True)

    # Draw samples
    ts = Gamma(
        'ts', alpha=np.float32(n / (2 * beta)), beta=np.float32(.5), 
        shape=n_samples, dtype=floatX
    )[:, np.newaxis]
    mus_ = Deterministic(
        'mus_', ts**(np.float32(0.5 / beta)) * us.dot(L_cov_)
    )
    mu1s_ = mus_[:, 0]
    mu2s_ = mus_[:, 1]

    if 10 <= verbose:
        print('GG for individual effect')
        print('gs.dtype = {}'.format(gs.dtype))
        print('ss.dtype = {}'.format(ss.dtype))
        print('ns.dtype = {}'.format(ns.dtype))
        print('us.dtype = {}'.format(us.dtype))
        print('ts.dtype = {}'.format(ts.dtype))

    return mu1s_, mu2s_


def _indvdl(hparams, std_x, n_samples, L_cov):
    dist_indvdl = hparams.dist_indvdl

    if dist_indvdl == 't':
        mu1s, mu2s = _indvdl_t(hparams, std_x, n_samples, L_cov)

    elif dist_indvdl == 'gauss':
        mu1s, mu2s = _indvdl_gauss(hparams, std_x, n_samples, L_cov)

    elif dist_indvdl == 'gg':
        mu1s, mu2s = _indvdl_gg(hparams, std_x, n_samples, L_cov)

    return mu1s, mu2s


def _get_L_cov(hparams):
    dist_l_cov_21 = hparams.dist_l_cov_21
    l_cov_21 = _dist_from_str('l_cov_21', dist_l_cov_21)
    l_cov = tt.stack([1.0, l_cov_21, l_cov_21, 1.0]).reshape((2, 2))

    return l_cov


def _noise_variance(hparams, tau_cmmn, verbose=0):
    dist_std_noise = hparams.dist_std_noise

    if dist_std_noise == 'tr_normal':
        h1 = HalfNormal('h1', tau=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = HalfNormal('h2', tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Truncated normal for prior scales')

    elif dist_std_noise == 'log_normal':
        h1 = Lognormal('h1', tau=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = Lognormal('h2', tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Log normal for prior scales')

    elif dist_std_noise == 'uniform':
        h1 = Uniform('h1', upper=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = Uniform('h2', upper=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Uniform for prior scales')

    else:
        raise ValueError(
            "Invalid value of dist_std_noise: %s" % dist_std_noise
        )

    return h1, h2


def _common_interceptions(hparams, tau_cmmn, verbose=0):
    prior_var_mu = hparams.prior_var_mu
    fix_mu_zero = hparams.fix_mu_zero

    if fix_mu_zero:
        mu1 = np.float32(0.0)
        mu2 = np.float32(0.0)

        if 10 <= verbose:
            print('Fix bias parameters to 0.0')

    else:
        if prior_var_mu == 'auto':
            tau1 = np.float32(1. / tau_cmmn[0])
            tau2 = np.float32(1. / tau_cmmn[1])
        else:
            v = prior_var_mu
            tau1 = np.float32(1. / v)
            tau2 = np.float32(1. / v)
        mu1 = Normal('mu1', mu=np.float32(0.), tau=np.float32(tau1), 
                     dtype=floatX)
        mu2 = Normal('mu2', mu=np.float32(0.), tau=np.float32(tau2), 
                     dtype=floatX)

        if 10 <= verbose:
            print('mu1.dtype = {}'.format(mu1.dtype))
            print('mu2.dtype = {}'.format(mu2.dtype))

    return mu1, mu2


def _noise_model(hparams, h1, h2):
    u"""Distribution of observation noise.
    """
    dist_noise = hparams.dist_noise

    if dist_noise == 'laplace':
        def obs1(mu):
            return _laplace_loglike(mu=mu, b=h1 / np.float32(np.sqrt(2.)))

        def obs2(mu):
            return _laplace_loglike(mu=mu, b=h2 / np.float32(np.sqrt(2.)))

    elif dist_noise == 'gg':
        dist_beta_noise = hparams.dist_beta_noise
        beta_noise = _dist_from_str('beta_noise', dist_beta_noise)

        def obs1(mu):
            return _gg_loglike(mu=mu, beta=beta_noise, std=h1)

        def obs2(mu):
            return _gg_loglike(mu=mu, beta=beta_noise, std=h2)

    else:
        raise ValueError(
            "Invalid value of dist_noise: %s" % dist_noise
        )

    return obs1, obs2


def _causal_model(hparams, w1_params, w2_params, tau_cmmn, d):
    subtract_mu_reg = hparams.subtract_mu_reg
    mu1, mu1s, obs1 = w1_params
    mu2, mu2s, obs2 = w2_params

    def likelihood(xs):
        # Independent variable
        obs1_ = obs1(mu=mu1 + mu1s)

        # Regression coefficient
        b = Normal('b%s' % d, mu=np.float32(0.), 
                   tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        # Dependent variable
        obs2_ = obs2(mu=mu2 + mu2s + b * (xs[:, 0] - mu1 - mu1s)) \
                if subtract_mu_reg else \
                obs2(mu=mu2 + mu2s + b * xs[:, 0])

        return obs1_(xs[:, 0]) + obs2_(xs[:, 1])

    return likelihood


def get_mixbml_model(xs, hparams, verbose=0):
    u"""Returns a PyMC3 probabilistic model of mixture BML.

    This function should be invoked within a Model session of PyMC3.

    :param xs: Observation data. 
    :type xs: ndarray, shape=(n_samples, 2), dtype=float
    :param hparams: Hyperparameters for inference.
    :type hparams: MixBMLParams
    :return: Probabilistic model 
    :rtype: pymc3.Model
    """
    # Standardize samples
    floatX = 'float32' # TODO: remove literal
    n_samples = xs.shape[0]
    xs = xs.astype(floatX)
    xs = standardize_samples(xs, True) if hparams.standardize else xs

    # Common scaling parameters
    std_x = np.std(xs, axis=0).astype(floatX)
    max_c = 1.0 # TODO: remove literal
    tau_cmmn = np.array(
        [(std_x[0] * max_c)**2, (std_x[1] * max_c)**2]).astype(floatX)

    # Prior of individual specific effects (\tilde{\mu}_{l}^{(i)})
    L_cov = _get_L_cov(hparams)
    mu1s, mu2s = _indvdl(hparams, std_x, n_samples, L_cov)

    # Noise variance
    h1, h2 = _noise_variance(hparams, tau_cmmn)

    # Common interceptions
    mu1, mu2 = _common_interceptions(hparams, tau_cmmn)

    # Noise model
    # obs1 (obs2) is a log likelihood function, not RV
    obs1, obs2 = _noise_model(hparams, h1, h2)

    # Pair of causal models
    v1_params = [mu1, mu1s, obs1]
    v2_params = [mu2, mu2s, obs2]

    # lp_m1: x1 -> x2 (b_21 is non-zero)
    # lp_m2: x2 -> x1 (b_12 is non-zero)
    lp_m1 = _causal_model(hparams, v1_params, v2_params, tau_cmmn, '21')
    lp_m2 = _causal_model(hparams, v2_params, v1_params, tau_cmmn, '12')

    # Prior of mixing proportions for causal models
    p = Beta('p', alpha=1, beta=1)

    # Mixture of potentials of causal models
    def lp_mix(xs):
        def flip(xs):
            # Filp 1st and 2nd features
            return tt.stack([xs[:, 1], xs[:, 0]], axis=0).T

        return logsumexp(tt.stack([tt.log(p) + lp_m1(xs),
                                   tt.log(1 - p) + lp_m2(flip(xs))], axis=0))

    DensityDist('dist', lp_mix, observed=xs)
