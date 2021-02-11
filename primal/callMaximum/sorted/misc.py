"""
This file implements the following functions:
    project_constant_to_zero():            Projects the constant coefficient of the input tensor to zero.
"""

import numpy as np
from scipy.linalg import sqrtm
import xerus as xe


################################################################################
#    Functions                                                                 #
################################################################################

def compute_discounted_payoff_PutAmer(num_assets,
                                      num_steps,
                                      num_samples,
                                      spot = np.array([1.0]),
                                      strike = 1.1,
                                      maturity = 1.0,
                                      trend = np.array([0.0]),
                                      volatility = np.array([0.2]),
                                      correlation = 0.0,
                                      interest = 0.5,
                                      rng = 0):
    """
    Draws samples of the discounted payoff for an american put option.

    Returns
    -------
    np.ndarray, np.ndarray
        Gaussian increments and paths.
    """
    # print('shapes', spot.shape, trend.shape, volatility.shape, (num_assets,))
    assert spot.shape == trend.shape == volatility.shape == (num_assets,)
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    else:
        assert isinstance(rng, np.random._generator.Generator)
    payoff_function = lambda S: np.maximum(0, strike-S)  # (premia: PutAmer) -- payoff function of an american put option
    dt = maturity/num_steps
    ts = dt*np.arange(num_steps+1)

    Gamma = np.full((num_assets, num_assets), correlation)
    i = np.arange(num_assets)
    Gamma[i,i] = 1
    L = volatility[:,None]*sqrtm(Gamma)

    parameters = rng.standard_normal((num_samples, num_assets, num_steps))  # centered normalized gaussian RVs

    B = np.empty((num_samples, num_assets, num_steps+1), dtype=float)
    B[...,0]  = 0
    B[...,1:] = np.sqrt(dt)*parameters
    B = np.cumsum(B, axis=-1)
    # transfomed_B = np.einsum('ab,nbs -> nas', L, B)
    transfomed_B = np.tensordot(L, B, axes=(1, 1)).transpose((1, 0, 2))
    diffusion = np.sum(L**2, axis=1)
    asset_value = spot[None,:,None] * np.exp(((np.ones(num_assets)*interest - trend)-diffusion/2)[None,:,None]*ts[None,None,:] + transfomed_B)

    assert asset_value.shape == (num_samples, num_assets, num_steps+1) and parameters.shape == (num_samples, num_assets, num_steps)
    basket_value = np.mean(asset_value, axis=1)
    assert basket_value.shape == (num_samples, num_steps+1)
    discounted_payoff = np.exp(-interest*ts)*payoff_function(basket_value)
    assert discounted_payoff.shape == (num_samples, num_steps+1)

    return parameters, asset_value, discounted_payoff


