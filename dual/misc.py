"""
This file implements the following functions:
    project_constant_to_zero():            Projects the constant coefficient of the input tensor to zero.
"""
from itertools import product
from math import comb

import numpy as np
from numpy.polynomial.hermite_e import hermeval
from scipy.linalg import sqrtm
import xerus as xe


################################################################################
#    Functions                                                                 #
################################################################################

def compute_asset_values(num_samples, num_assets, num_exercise_dates, spot, volatility, correlation, maturity, interest, dividends, rng=None):
    """
    Draws asset value samples from the Black-Scholes model.

    Returns
    -------
    np.ndarray, np.ndarray
        Gaussian increments and asset values.
    """
    assert spot.shape == volatility.shape == dividends.shape == (num_assets,)
    if rng is None or isinstance(rng, int): rng = np.random.default_rng(rng)
    assert isinstance(rng, np.random._generator.Generator)
    dt = maturity/(num_exercise_dates-1)
    ts = dt*np.arange(num_exercise_dates)

    Gamma = np.full((num_assets, num_assets), correlation)
    i = np.arange(num_assets)
    Gamma[i,i] = 1
    L = np.linalg.cholesky(Gamma)
    assert np.allclose(L@L.T, Gamma)

    increments = rng.standard_normal((num_samples, num_assets, num_exercise_dates-1))  # centered normalized gaussian RVs
    assert increments.shape == (num_samples, num_assets, num_exercise_dates-1)

    B = np.empty((num_samples, num_assets, num_exercise_dates), dtype=float)
    B[...,0]  = 0
    B[...,1:] = np.sqrt(dt)*increments
    B = np.cumsum(B, axis=-1)
    B = np.einsum('ab,nbs -> nas', L.T, B)
    asset_values = spot[None,:,None] * np.exp((interest-dividends-volatility**2/2)[None,:,None]*ts[None,None,:] + volatility[None,:,None]*B)
    assert asset_values.shape == (num_samples, num_assets, num_exercise_dates)

    return increments, asset_values


def CallMaximumPayoff(strike):  # (premia: CallMaximumAmer_nd) -- payoff function of a max call put option
    return lambda asset_values: np.maximum(0, np.max(asset_values, axis=1)-strike)


def PutBasketPayoff(strike):  # (premia: PutBasketAmer_nd) -- payoff function of an american put option
    return lambda asset_values: np.maximum(0, strike-np.mean(asset_values, axis=1))


def compute_discounted_payoffs(asset_values, payoff_function, maturity, interest):
    """
    Computes the discounted payoff given the asset values and a payoff function.

    Returns
    -------
    np.ndarray
        Payoffs.
    """
    num_samples, num_assets, num_exercise_dates = asset_values.shape
    dt = maturity/(num_exercise_dates-1)
    ts = dt*np.arange(num_exercise_dates)

    discounted_payoff = np.exp(-interest*ts)*payoff_function(asset_values)
    assert discounted_payoff.shape == (num_samples, num_exercise_dates)

    return discounted_payoff


def multiIndices(_degree, _order):
    return filter(lambda mI: sum(mI) <= _degree, product(range(_degree+1), repeat=_order))


def hermite_measures(_points, _degree):
    """
    Evaluation of the first p=_dimension Hermite polynomials `[He_0(y), He_1(y), ..., He_{p-1}(y)]` on each point `y` in `_points`.
    """
    num_samples, num_assets, num_steps = _points.shape
    tmp = hermeval(_points, np.eye(_degree+1))  #NOTE: The Hermite polynomials are NOT normalized.
    assert tmp.shape == (_degree+1, num_samples, num_assets, num_steps)
    mIs = list(multiIndices(_degree, num_assets))
    assert len(mIs) == comb(_degree+num_assets, num_assets)
    j = np.arange(num_assets)
    ret = np.empty((len(mIs), num_samples, num_steps))
    for i,mI in enumerate(mIs):
        ret[i] = np.prod(tmp[mI,:,j,:], axis=0)
    return ret.T
