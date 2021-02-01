"""
Minimizes a function ∑ₖf(T,xₖ,yₖ) ≈ E[f(T,•,•)] over the set of low-rank manifolds using a Riemannian gradient descent scheme.
The expectation E[f] is convex and almost surely differentiable.

The one important aspect of the optimization is that T is constraint to have T[0]=0.
The initial value is chosen to satisfy this requirement and the gradients are projected to honour it.

This file implements the following functions:
    costs():                               Computes Monte-Carlo estimates of the first moment E[f(T,•,•)] and the second moment E[f(T,•,•)²].
    costs_component_gradient():            Computes the gradient of `costs` with respect to a specific component tensor.
    costs_riemannian_gradient():           Computes the Riemannian gradient of `costs`.
    retraction():                          Computes an approximation of the exponential map.
"""
import autoPDB

import os
import pickle
from collections import deque, namedtuple
from multiprocessing import Pool, cpu_count
from time import process_time
from hashlib import md5
import json
from math import comb
from shutil import rmtree
from functools import wraps

import numpy as np
import xerus as xe

from misc import compute_asset_values, compute_discounted_payoffs, PutBasketPayoff, CallMaximumPayoff, hermite_measures, multiIndices
from riemann import *


################################################################################
#    Functions                                                                 #
################################################################################

def maxDiffSample(_i, _tt, _measures, _values, _alpha, _leftStack0, _rightStack):
    dimensions = _tt.dimensions
    num_steps = len(dimensions)
    leftStackY = [None]*(num_steps+1)
    leftStackY[0] = np.ones((1,))
    for m in range(1,num_steps+1):
        leftStackY[m] = np.einsum('l,ler,e -> r', leftStackY[m-1], _tt.get_component(m-1).to_ndarray(), _measures[m-1][_i,:dimensions[m-1]])

    ds = np.empty(num_steps+1)
    for m in range(num_steps+1):  # loop time steps
        condExpProj = (leftStackY[m] - _leftStack0[m]) @ _rightStack[m]
        ds[m] = _values[_i,m] - condExpProj

    eads = np.exp(_alpha*(ds - np.max(ds)))
    maxDiff = np.sum(ds*eads) / np.sum(eads)

    return maxDiff


def maxDiffSegments(_args):
    _start, _stop, _tt, _measures, _values, _alpha, _leftStack0, _rightStack = _args
    moment1 = 0
    moment2 = 0
    for i in range(_start, _stop):
        mds = maxDiffSample(i, _tt, _measures, _values, _alpha, _leftStack0, _rightStack)
        moment1 += mds
        moment2 += mds**2
    return moment1, moment2


def costs(_tt, _measures, _values, _alpha=50):
    """
    Compute Monte-Carlo estimates of the first moment E[f(T,•,•)] and the second moment E[f(T,•,•)²].

    Parameters
    ----------
    _tt : xe.TTTensor
        The tensor T for which the costs are computed.
    _measures, _values : np.ndarray
        The samples xₖ and yₖ used for Monte-Carlo estimation.
    _alpha : float
        Smoothness parameter for the smooth maximum function. (default: 1.0)

    Returns
    -------
    float, float
        The first and second moments.
    """
    _num_steps = _tt.order()
    assert len(_measures) == _num_steps
    _num_samples = _measures[0].shape[0]
    dimensions = _tt.dimensions
    for m in range(_num_steps):
        assert _measures[m].ndim == 2 and _measures[m].shape[0] == _num_samples and _measures[m].shape[1] >= dimensions[m]
    assert _values.shape == (_num_samples, _num_steps+1)

    assert isinstance(_measures, np.ndarray)
    measures = _measures.view()
    measures.flags.writeable = False
    values = _values.view()
    values.flags.writeable = False

    leftStack0 = [None]*(_num_steps+1)
    leftStack0[0] = np.ones((1,))
    leftStack0[0].flags.writeable = False
    for m in range(1,_num_steps+1):
        leftStack0[m] = leftStack0[m-1] @ _tt.get_component(m-1).to_ndarray()[:,0,:]
        leftStack0[m].flags.writeable = False

    rightStack = [None]*(_num_steps+1)
    rightStack[_num_steps] = np.ones((1,))
    rightStack[_num_steps].flags.writeable = False
    for m in reversed(range(_num_steps)):
        rightStack[m] = _tt.get_component(m).to_ndarray()[:,0,:] @ rightStack[m+1]
        rightStack[m].flags.writeable = False

    cpucount = cpu_count()
    chunksize = _num_samples // cpucount + (_num_samples % cpucount != 0)
    assert chunksize * cpucount >= _num_samples
    arguments = [(x*chunksize, min((x+1)*chunksize, _num_samples), xe.TTTensor(_tt), measures, values, _alpha, leftStack0, rightStack) for x in range(cpucount)]
    with Pool() as p:
        maxDiffs = p.map(maxDiffSegments, arguments, chunksize=1)
    maxDiffs = np.sum(maxDiffs, axis=0) / _num_samples
    assert maxDiffs.shape == (2,)
    return maxDiffs


def maxDiffGradSample(_i, _tt, _mode, _measures, _values, _alpha, _leftStack0, _rightStack):
    dimensions = _tt.dimensions
    num_steps = len(dimensions)
    leftStackY = [None]*(num_steps+1)
    leftStackY[0] = np.ones((1,))
    for m in range(1,num_steps+1):
        leftStackY[m] = np.einsum('l,ler,e -> r', leftStackY[m-1], _tt.get_component(m-1).to_ndarray(), _measures[m-1][_i,:dimensions[m-1]])

    ds = np.empty(num_steps+1)
    for m in range(num_steps+1):  # loop time steps
        condExpProj = (leftStackY[m] - _leftStack0[m]) @ _rightStack[m]
        ds[m] = _values[_i,m] - condExpProj

    eads = np.exp(_alpha*(ds - np.max(ds)))
    seads = np.sum(eads)
    maxDiff = np.sum(ds*eads) / seads

    # condExpProj at time step m = (leftStackY[m] - leftStack0[m]) @ rightStack[m]
    dgs = np.empty([num_steps+1]+_tt.get_component(_mode).dimensions)
    for m in range(_mode+1):
        # Differentiation happens in one of the factors of rightStack[m].
        # In lFactor all components left of (but excluding) coeffsTT[_mode] are measured.
        lFactor = leftStackY[m] - _leftStack0[m]
        for mm in range(m, _mode):
            lFactor = np.einsum('l,lr -> r', lFactor, _tt.get_component(mm).to_ndarray()[:,0,:])
        # cFactor measures the component coeffsTT[_mode].
        cFactor = np.eye(1,dimensions[_mode])[0]
        # In rFactor all components right of (and including) coeffsTT[_mode+1] are measured.
        rFactor = _rightStack[_mode+1]
        dgs[m] = -np.einsum('l,e,r -> ler', lFactor, cFactor, rFactor)
    for m in range(_mode+1, num_steps+1):
        # Differentiation happens in one of the factors of leftStack[m].
        # In lFactor all components left of (but excluding) coeffsTT[_mode] are measured.
        lFactorY = leftStackY[_mode]
        lFactor0 = _leftStack0[_mode]
        # cFactor measures the component coeffsTT[_mode].
        cFactorY = _measures[_mode][_i,:dimensions[_mode]]
        cFactor0 = np.eye(1,dimensions[_mode])[0]
        # In rFactor all components right of (and including) coeffsTT[_mode+1] are measured.
        rFactorY = _rightStack[m]
        for mm in reversed(range(_mode+1, m)):
            rFactorY = np.einsum('e,ler,r -> l', _measures[mm][_i,:dimensions[mm]], _tt.get_component(mm).to_ndarray(), rFactorY)
        rFactor0 = _rightStack[_mode+1]
        dgs[m] = -(np.einsum('l,e,r -> ler', lFactorY, cFactorY, rFactorY) - np.einsum('l,e,r -> ler', lFactor0, cFactor0, rFactor0))

    gradient = np.zeros(_tt.get_component(_mode).dimensions)
    for m in range(num_steps+1):  # loop time steps
        gradient += eads[m]/seads * (1 + _alpha*(ds[m] - maxDiff)) * dgs[m]

    return gradient


def maxDiffGradSegments(_args):
    _start, _stop, _tt, _mode, _measures, _values, _alpha, _leftStack0, _rightStack = _args
    gradSum = 0
    for i in range(_start, _stop):
        gradSum += maxDiffGradSample(i, _tt, _mode, _measures, _values, _alpha, _leftStack0, _rightStack)
    return gradSum


def costs_component_gradient(_tt, _mode, _measures, _values, _alpha=50):
    """
    Compute the gradient of the cost functional with respect to the `_mode`s component tensor.

    Parameters
    ----------
    _tt : xe.TTTensor
        The point at which the gradient is to be computed.
    _mode : int
        The index of the component tensor with respect to which the gradient is to be computed.
    _measures, _values : np.ndarray
        The samples used for Monte-Carlo estimation.
    _alpha : float
        Smoothness parameter for the smooth maximum function. (default: 1.0)

    Returns
    -------
    np.ndarray
        The gradient.

    Notes
    -----
        The correctness of this function has been verified via a Taylor test.
    """
    _num_steps = _tt.order()
    assert len(_measures) == _num_steps
    _num_samples = _measures[0].shape[0]  # num_samples
    dimensions = _tt.dimensions
    for m in range(_num_steps):
        assert _measures[m].ndim == 2 and _measures[m].shape[0] == _num_samples and _measures[m].shape[1] >= dimensions[m]
    assert _values.shape == (_num_samples, _num_steps+1)
    assert 0 <= _mode < _num_steps

    assert isinstance(_measures, np.ndarray)
    measures = _measures.view()
    measures.flags.writeable = False
    values = _values.view()
    values.flags.writeable = False

    leftStack0 = [None]*(_num_steps+1)
    leftStack0[0] = np.ones((1,))
    leftStack0[0].flags.writeable = False
    for m in range(1,_num_steps+1):
        leftStack0[m] = leftStack0[m-1] @ _tt.get_component(m-1).to_ndarray()[:,0,:]
        leftStack0[m].flags.writeable = False

    rightStack = [None]*(_num_steps+1)
    rightStack[_num_steps] = np.ones((1,))
    rightStack[_num_steps].flags.writeable = False
    for m in reversed(range(_num_steps)):
        rightStack[m] = _tt.get_component(m).to_ndarray()[:,0,:] @ rightStack[m+1]
        rightStack[m].flags.writeable = False

    cpucount = cpu_count()
    chunksize = _num_samples // cpucount + (_num_samples % cpucount != 0)
    assert chunksize * cpucount >= _num_samples
    arguments = [(x*chunksize, min((x+1)*chunksize, _num_samples), xe.TTTensor(_tt), _mode, measures, values, _alpha, leftStack0, rightStack) for x in range(cpucount)]
    with Pool() as p:
        maxDiffGrad = p.map(maxDiffGradSegments, arguments, chunksize=1)
    maxDiffGrad = np.sum(maxDiffGrad, axis=0) / _num_samples
    assert maxDiffGrad.shape == tuple(_tt.get_component(_mode).dimensions)
    return maxDiffGrad


def costs_component_gradient_fd(_tt, _mode, _measures, _values, _h=1e-8):
    val0 = costs(_tt, _measures,  _values)[0]
    test = xe.TTTensor(_tt)
    ret = xe.Tensor(_tt.get_component(_mode).dimensions)
    for I in range(ret.size):
        testCore = xe.Tensor(_tt.get_component(_mode))
        testCore[I] += _h
        test.set_component(_mode, testCore)
        valI = costs(test, _measures, _values)[0]
        ret[I] = (valI - val0) / _h
    return ret.to_ndarray()


def costs_riemannian_gradient(_tt, _measures, _values):
    """
    Computes the Riemannian gradient of the cost functional.

    Parameters
    ----------
    _tt : xe.TTTensor
        The point at which the gradient is to be computed.
    _measures, _values : np.ndarray
        The samples used for Monte-Carlo estimation.

    Returns
    -------
    TangentVector
        The gradient.
    """

    #TODO: das geht schneller, wenn man Us, Vs und Xs speichert und den TangentSpace manuell aufbaut.
    gradient = xe.TTTensor(_tt.dimensions)
    _tt.move_core(0)
    for m in range(_tt.order()):
        core = costs_component_gradient(_tt, m, _measures, _values)
        cg = xe.TTTensor(_tt)
        if m < _tt.order()-1:
            _tt.move_core(m+1)
            Um = _tt.get_component(m).to_ndarray()
            core -= np.einsum('lex, yzx, yzr -> ler', Um, Um, core)
        cg.set_component(m, xe.Tensor.from_buffer(core))
        gradient = gradient + cg

    ts = TangentSpace(_tt)
    tv = ts.project(gradient)
    assert xe.frob_norm(tv.to_TTTensor() - gradient) <= 1e-12*xe.frob_norm(gradient)

    # project out the part in direction [0, ..., 0]
    tp = ts.project(xe.TTTensor.dirac(_tt.dimensions, [0]*_tt.order()))
    tv = (tv - (tv @ tp) / tp.norm()**2 * tp)
    return tv


def retraction(_tt, _tv, _roundingParameter):
    """
    Compute an approximation of the exponential map.

    Parameters
    ----------
    _tv : TangentVector
        This object contains both the starting point and the direction of the line.
    _roundingEps: float
        Accuracy used for rounding.

    Returns
    -------
    callable
        A function taking a step size and returning a TTTensor.
    """
    basePoint = TangentVector(_tt)
    tv_norm = _tv.norm()
    def step(_stepSize):
        trial = (basePoint - _stepSize*_tv).to_TTTensor()
        tmp = xe.TTTensor(trial)
        trial.round(_roundingParameter)
        return trial, xe.frob_norm(tmp-trial)/(_stepSize*tv_norm)
    return step


class Parameters(object):
    __slots__ = ["num_training_samples", "num_validation_samples", "num_assets", "num_exercise_dates",
                 "spot", "volatility", "correlation", "maturity", "interest", "dividends", "option", "strike",
                 "chaos_degree", "chaos_rank"]

    def __init__(self, num_training_samples,
                       num_validation_samples,
                       num_assets,
                       num_exercise_dates,
                       spot,
                       volatility,
                       correlation,
                       maturity,
                       interest,
                       dividends,
                       option,
                       strike,
                       chaos_degree,
                       chaos_rank):

        self.num_training_samples = num_training_samples
        self.num_validation_samples = num_validation_samples
        self.num_assets = num_assets
        self.num_exercise_dates = num_exercise_dates
        self.spot = spot
        self.volatility = volatility
        self.correlation = correlation
        self.maturity = maturity
        self.interest = interest
        self.dividends = dividends
        self.option = option
        self.strike = strike
        self.chaos_degree = chaos_degree
        self.chaos_rank = chaos_rank

    def __iter__(self):
        return (getattr(self, slot) for slot in self.__slots__)

    def __hash__(self):
        hsh = md5()
        data = [np.asarray(item).tolist() for item in self]
        hsh.update(json.dumps(data, sort_keys=True).encode('utf-8'))
        return int(hsh.hexdigest(), 16)

    def __eq__(self, other):
        return all(np.all(es == eo) for es,eo in zip(self, other))

    def modify(self, **kwargs):
        ret = Parameters(*self)
        for key,val in kwargs.items():
            setattr(ret, key, val)
        return ret

    def copy(self): return self.modify()


Solution = namedtuple("Solution", ["parameters", "value", "time"])


def cache(function):
    # def hash_ndarray(_arr):
    #     hsh = md5()
    #     hsh.update(np.ascontiguousarray(_arr).data)
    #     return hsh.hexdigest()

    def hash_tt(_tt):
        hsh = md5()
        hsh.update(np.array(_tt.dimensions).data)
        hsh.update(np.array(_tt.ranks()).data)
        for m in range(_tt.order()):
            core = _tt.get_component(m).to_ndarray()
            hsh.update(core.data)
        return hsh.hexdigest()

    def hash_obj(_obj):
        assert isinstance(_obj, (xe.TTTensor, tuple))
        if isinstance(_obj, xe.TTTensor): return hash_tt(_obj)
        assert len(_obj) == 2 and isinstance(_obj[0], np.ndarray) and isinstance(_obj[1], np.ndarray)
        hsh = md5()
        hsh.update(np.ascontiguousarray(_obj[0]).data)
        hsh.update(np.ascontiguousarray(_obj[1]).data)
        return hsh.hexdigest()

    @wraps(function)
    def cached_function(params, *args, **kwargs):
        baseName = f".cache/american_option_tt_{hash(params)}"
        os.makedirs(baseName, exist_ok=True)
        try:
            with open(f"{baseName}/{function.__name__}.pkl", "rb") as f:
                sol = pickle.load(f)
            assert sol.parameters == params
            print(f"Loading: {hash_obj(sol.value)}")
            return sol
        except FileNotFoundError:
            print("="*80)
            print(f"  Calling '{function.__name__}'")
            print("="*80)
            tic = process_time()
            value = function(params, *args, **kwargs)
            toc = process_time()
            time = toc-tic
            sol = Solution(params, value, time)
            print(f"Caching: {hash_obj(value)}")
            with open(f"{baseName}/{function.__name__}.pkl", "wb") as f:
                pickle.dump(sol, f, pickle.HIGHEST_PROTOCOL)
        return sol
    return cached_function


def clear_cache(params):
    baseName = f".cache/american_option_tt_{hash(params)}"
    if os.path.exists(baseName):
        rmtree(baseName)


def is_cached(params, function):
    baseName = f".cache/american_option_tt_{hash(params)}"
    return os.path.exists(f"{baseName}/{function.__name__}.pkl")


@cache
def compute_and_cache_measures_and_values(parameters):
    num_samples = parameters.num_training_samples + parameters.num_validation_samples
    if parameters.option == "PutBasket":
        payoff_function = PutBasketPayoff(parameters.strike)
    elif parameters.option == "CallMaximum":
        payoff_function = CallMaximumPayoff(parameters.strike)

    increments, asset_values = compute_asset_values(num_samples        = num_samples,
                                                    num_assets         = parameters.num_assets,
                                                    num_exercise_dates = parameters.num_exercise_dates,
                                                    spot               = parameters.spot,
                                                    volatility         = parameters.volatility,
                                                    correlation        = parameters.correlation,
                                                    maturity           = parameters.maturity,
                                                    interest           = parameters.interest,
                                                    dividends          = parameters.dividends,
                                                    rng                = 0)
    payoffs = compute_discounted_payoffs(asset_values, payoff_function, parameters.maturity, parameters.interest)
    assert payoffs.shape == (num_samples, parameters.num_exercise_dates)
    measures = hermite_measures(increments, parameters.chaos_degree)
    assert measures.shape == (parameters.num_exercise_dates-1, num_samples, comb(parameters.chaos_degree+parameters.num_assets, parameters.num_assets))
    return measures, payoffs


@cache
def compute_and_cache_initial_guess(params, maxIter=100):
    # order = params.num_exercise_dates-1
    # dimension = comb(params.chaos_degree+params.num_assets, params.num_assets)
    # tt = xe.TTTensor.random([dimension]*order, [params.chaos_rank]*(order-1))
    # return 1e-6*tt/xe.frob_norm(tt)

    # measures, values = compute_and_cache_measures_and_values(params).value
    # num_steps, num_samples, chaos_dimension = measures.shape
    # assert values.shape == (num_samples, num_steps+1)

    # values = values[:,-1:]  # just take the last value
    # values = [xe.Tensor.from_ndarray(val) for val in values]
    # measures = np.transpose(measures, axes=(1,0,2))
    # measures = [[xe.Tensor.from_ndarray(cmp_m) for cmp_m in m] for m in measures]

    # tt = xe.uq_ra_adf(measures, values, [1]+[chaos_dimension]*num_steps, targeteps=1e-6, maxitr=maxIter)
    # tt.fix_mode(0,0)  # remove the physical dimension (1) from reconstruction
    # tt.round(1e-6)    # remove unnecessary ranks
    # return tt

    order = params.num_exercise_dates-1
    assert params.chaos_degree >= 1
    assert order >= 1

    if params.chaos_degree == 1:
        return xe.TTTensor([1+params.num_assets]*order)

    paramsRec = params.copy()
    paramsRec.chaos_degree -= 1
    ttRec = compute_and_cache_solution(paramsRec, maxIter).value

    dimension = comb(params.chaos_degree+params.num_assets, params.num_assets)
    dimensionRec = comb(paramsRec.chaos_degree+paramsRec.num_assets, paramsRec.num_assets)
    assert ttRec.dimensions == [dimensionRec]*order

    mIs = list(multiIndices(params.chaos_degree,   params.num_assets))      # multi-indices for polynomials of degree params.chaos_degree
    mIRecs = list(multiIndices(paramsRec.chaos_degree, params.num_assets))  # multi-indices for polynomials of degree params.chaos_degree-1
    assert len(mIRecs) == dimensionRec and len(mIs) == dimension

    tt = xe.TTTensor([dimension]*order)
    for m in range(order):
        coreRec = ttRec.get_component(m).to_ndarray()
        assert coreRec.shape[1] == dimensionRec
        core = np.empty((coreRec.shape[0], dimension, coreRec.shape[2]))
        iRec = 0
        for i,mI in enumerate(mIs):
            if sum(mI) < params.chaos_degree:  # mI is also an element of mIRecs
                assert mIRecs[iRec] == mI      # ensure that the ordering of the elements of mIRecs in mIs is the same as in mIRecs
                core[:,i,:] = coreRec[:,iRec,:]
                iRec += 1
            else:                              # mI is a new multi-index
                core[:,i,:] = 0
        assert iRec == dimensionRec            # ensure that every multi-index was used
        tt.set_component(m, xe.Tensor.from_ndarray(core))
    tt.move_core(0)

    # #TODO: kick out of local minimum
    # rk = tt.ranks()
    # kick = xe.TTTensor.random(tt.dimensions, rk)
    # tt = tt + 0.1*xe.frob_norm(tt)/xe.frob_norm(kick)*kick
    # tt.round(rk)

    return tt


@cache
def compute_and_cache_solution(params, maxIter=100):
    trainingSet   = slice(0, params.num_training_samples, 1)
    validationSet = slice(params.num_training_samples, params.num_training_samples+params.num_validation_samples, 1)

    sol = compute_and_cache_measures_and_values(params)
    measures, values = sol.value
    time = sol.time

    sol = compute_and_cache_initial_guess(params, maxIter)
    tt = sol.value
    time += sol.time

    ranks = [params.chaos_rank]*(params.num_exercise_dates-2)

    # Define convenience functions.
    def training_costs(_tt):
        return costs(_tt, measures[:, trainingSet], values[trainingSet])[0]

    def training_costs_gradient(_tt):
        return costs_riemannian_gradient(_tt, measures[:, trainingSet], values[trainingSet])

    def validation_costs(_tt):
        return costs(_tt, measures[:, validationSet], values[validationSet], _alpha=1e3)[0]

    # compute_descentDir = lambda curGrad, prevGrad: curGrad  # GD
    # compute_descentDir = lambda curGrad, prevGrad: 0.5*(curGrad + prevGrad)  # Momentum
    def compute_descentDir(curGrad, prevGrad):  # nonlinear CG update
        """
        Nonlinear CG update.

        The Hestenes-Stiefel (HS) update can be derived by demanding that consecutive search directions be conjugate
        with respect to the average Hessian over the line segment [x_k , x_{k+1}].
        Even though it is a natural choice it is not easy to implement on Manifolds.
        The Polak-Ribiere (PR) update is similar to HS, both in terms of theoretical convergence properties and practical performance.
        For PR however, the strong Wolfe conditions does not guarantee that the computed update direction
        is always a descent direction. To guarantee this we modify PR to PR+. This choice also provides a direction reset automatically [2].
        Finally, it can be shown that global convergence can be guaranteed for every parameter that is bounded in absolute value by the Fletcher-Reeves update.
        This leads us to the final update rule max{PR+,FR}.
        To ensure that a descent direction is returned even with Armijo updates we check that the computed update direction
        does not point in the opposite direction to the gradient.

        References:
        -----------
          - [1] Numerical optimization (Jorge Nocedal and Stephen J. Wright)
          - [2] An Introduction to the Conjugate Gradient Method Without the Agonizing Pain (Jonathan Richard Shewchuk)
        """
        gradDiff = curGrad - prevGrad
        betaPR = (curGrad @ gradDiff) / (curGrad @ curGrad)    # Polak-Ribiere update
        beta = max(betaPR, 0)                                  # PR+ update
        betaFR = (curGrad @ curGrad) / (prevGrad @ prevGrad)   # Fletcher-Reeves update
        beta = min(beta, betaFR)                               # max{PR+,FR} update
        descentDir = curGrad + beta*prevGrad
        if descentDir @ curGrad < 1e-3 * descentDir.norm() * curGrad.norm():
            print("WARNING: Computed descent direction opposite to gradient.")
            descentDir = curGrad
        return descentDir

    print("="*80)
    print("  Perform gradient descent")
    print("="*80)
    tic = process_time()
    trnCosts = training_costs(tt)
    valCosts = deque(maxlen=10)
    grad = training_costs_gradient(tt)
    print(f"[0] Training costs: {trnCosts: .4e}  |  Validation costs: {validation_costs(tt): .4e}  |  Best validation costs: {np.nan: .4e}  |  Relative gradient norm: {grad.norm()/xe.frob_norm(tt):.2e}  |  Relative update norm: {np.nan:.2e}  |  Step size: {np.nan:.2e}  |  Relative retraction error: {np.nan:.2e}  |  Ranks: {tt.ranks()}")
    ss = 1
    descentDir = grad
    descentDirGrad = descentDir @ grad
    bestValCosts = np.inf
    bestTT = None
    for iteration in range(maxIter):
        if grad.norm() < 1e-6 * xe.frob_norm(tt):
            print("Termination: relative norm of gradient deceeds tolerance (local minimum reached)")
            break
        prev_tt = tt
        tt, re, ss = armijo_step(retraction(tt, descentDir, _roundingParameter=ranks), training_costs, descentDirGrad, _initialStepSize=ss)
        trnCosts = training_costs(tt)
        valCosts.append(validation_costs(tt))
        if valCosts[-1] < bestValCosts:
            bestTT = xe.TTTensor(tt)
            bestValCosts = valCosts[-1]
        print(f"[{iteration+1}] Training costs: {trnCosts: .4e}  |  Validation costs: {valCosts[-1]: .4e}  |  Best validation costs: {bestValCosts: .4e}  |  Relative gradient norm: {grad.norm()/np.asarray(xe.frob_norm(prev_tt)):.2e}  |  Relative update norm: {xe.frob_norm(prev_tt-tt)/np.asarray(xe.frob_norm(prev_tt)):.2e}  |  Step size: {ss:.2e}  |  Relative retraction error: {re:.2e}  |  Ranks: {tt.ranks()}")
        if len(valCosts) == 10 and (valCosts[0]-valCosts[-1]) < 1e-2*valCosts[0]:
            print("Termination: decrease of costs deceeds tolerance")
            break
        if iteration < maxIter-1:
            prev_grad = TangentSpace(tt).project(grad)
            grad = training_costs_gradient(tt)
            descentDir = compute_descentDir(grad, prev_grad)
            descentDirGrad = descentDir @ grad
    else:
        print("Termination: maximum number of iterations reached")

    assert bestTT is not None
    return bestTT


def compute_test_costs(parameters, numTestSamples, rng=None):
    if parameters.option == "PutBasket":
        payoff_function = PutBasketPayoff(parameters.strike)
    elif parameters.option == "CallMaximum":
        payoff_function = CallMaximumPayoff(parameters.strike)

    increments, asset_values = compute_asset_values(num_samples        = numTestSamples,
                                                    num_assets         = parameters.num_assets,
                                                    num_exercise_dates = parameters.num_exercise_dates,
                                                    spot               = parameters.spot,
                                                    volatility         = parameters.volatility,
                                                    correlation        = parameters.correlation,
                                                    maturity           = parameters.maturity,
                                                    interest           = parameters.interest,
                                                    dividends          = parameters.dividends)
    payoffs = compute_discounted_payoffs(asset_values, payoff_function, parameters.maturity, parameters.interest)
    assert payoffs.shape == (numTestSamples, parameters.num_exercise_dates)
    measures = hermite_measures(increments, parameters.chaos_degree)
    assert measures.shape == (parameters.num_exercise_dates-1, numTestSamples, comb(parameters.chaos_degree+parameters.num_assets, parameters.num_assets))

    assert is_cached(parameters, compute_and_cache_solution)
    tt = compute_and_cache_solution(parameters).value

    moment1, moment2 = costs(tt, measures, payoffs, _alpha=1e3)
    variance = max(moment2 - moment1**2, 0) * numTestSamples/(numTestSamples-1)  # the variance of the samples
    # Since the samples are independent (and thus uncorrelated) the variance of the mean estimator is given by Var(1/n sum(...)) = 1/n**2 sum(Var(...)) = 1/n**2 sum(variance) = n/n**2 variances = variance/n.
    return moment1, np.sqrt(variance/numTestSamples)


if __name__ == "__main__":
    CostsTest = True
    if CostsTest:
        print("Costs test:")
        print("-----------")
        from american_option_array import costs as costs_array

        num_exercise_dates = 6
        num_assets = 2
        degree = 1
        num_samples = 20000                              # number of samples used for the Monte-Carlo integration
        dimension = comb(degree+num_assets, num_assets)  # dimensions of the coefficient tensor
        rank = 3                                         # TT-ranks of the coefficient tensor
        spot = 1
        strike = 1.1

        def runCostsTest():
            print(f"Number of samples:        {num_samples:>{len(str(num_samples))}d}")
            print(f"Number of assets:         {num_assets:>{len(str(num_samples))}d}")
            print(f"Number of exercise dates: {num_exercise_dates:>{len(str(num_samples))}d}")
            print(f"Chaos degree:             {degree:>{len(str(num_samples))}d}")
            print(f"Dimension:                {dimension:>{len(str(num_samples))}d}")
            print(f"Ranks:                    {rank:>{len(str(num_samples))}d}")
            print(f"Spot:                     {spot:>{len(str(num_samples))}.1f}")
            print(f"Strike:                   {strike:>{len(str(num_samples))}.1f}")
            print()

            # Compute the samples for Monte-Carlo integration.
            increments, asset_values, values = compute_discounted_payoff_PutAmer(num_assets   = num_assets,
                                                                                 num_steps    = num_exercise_dates-1,
                                                                                 num_samples  = num_samples,
                                                                                 spot         = np.full(num_assets, spot),
                                                                                 strike       = strike,
                                                                                 trend        = np.zeros(num_assets),
                                                                                 volatility   = np.full(num_assets, 0.2),
                                                                                 correlation  = 0.2)

            measures = hermite_measures(increments, degree)
            assert values.shape == (num_samples, num_exercise_dates)
            assert measures.shape == (num_exercise_dates-1, num_samples, dimension)

            # Define a random initial value.
            tt = xe.TTTensor.random([dimension]*(num_exercise_dates-1), [rank]*(num_exercise_dates-2))
            arr = xe.Tensor(tt).to_ndarray()
            arr[(0,)*(num_exercise_dates-1)] = 0

            carr = costs_array(arr, measures, values)
            print(f"Costs[array]:     {carr[0]:.2e} \u00B1 {np.sqrt(carr[1]/num_samples):.2e}")
            for aexp in range(5):
                ctt = costs(tt, measures, values, 10**aexp)
                print(f"Costs[tt|\u03b1=1e{aexp:02d}]: {ctt[0]:.2e} \u00B1 {np.sqrt(ctt[1]/num_samples):.2e}")
                print(f"Errors:           {abs(carr[0]-ctt[0]):.2e} & {abs(np.sqrt(carr[1]/num_samples)-np.sqrt(ctt[1]/num_samples)):.2e}")
            print()

        runCostsTest()
        spot = 100
        strike = 100
        runCostsTest()

    TaylorTest = True
    if TaylorTest:
        print("Taylor test:")
        print("------------")

        num_exercise_dates = 6
        num_assets = 3
        degree = 2
        num_samples = 20000                              # number of samples used for the Monte-Carlo integration
        dimension = comb(degree+num_assets, num_assets)  # dimensions of the coefficient tensor
        rank = 3                                         # TT-ranks of the coefficient tensor
        print(f"Number of samples:        {num_samples:>{len(str(num_samples))}d}")
        print(f"Number of assets:         {num_assets:>{len(str(num_samples))}d}")
        print(f"Number of exercise dates: {num_exercise_dates:>{len(str(num_samples))}d}")
        print(f"Chaos degree:             {degree:>{len(str(num_samples))}d}")
        print(f"Dimension:                {dimension:>{len(str(num_samples))}d}")
        print(f"Ranks:                    {rank:>{len(str(num_samples))}d}")
        print()

        # Compute the samples for Monte-Carlo integration.
        increments, asset_values, values = compute_discounted_payoff_PutAmer(num_assets   = num_assets,
                                                                             num_steps    = num_exercise_dates-1,
                                                                             num_samples  = num_samples,
                                                                             spot         = np.ones(num_assets),
                                                                             trend        = np.zeros(num_assets),
                                                                             volatility   = np.full(num_assets,0.2),
                                                                             correlation  = 0.2)

        measures = hermite_measures(increments, degree)
        assert values.shape == (num_samples, num_exercise_dates)
        assert measures.shape == (num_exercise_dates-1, num_samples, dimension)

        # Define a random initial value.
        tt = xe.TTTensor.random([dimension]*(num_exercise_dates-1), [rank]*(num_exercise_dates-2))

        for hexp in range(3,10):
            err = 0
            for m in range(tt.order()):
                err += np.linalg.norm(
                            costs_component_gradient(tt, m, measures, values)
                          - costs_component_gradient_fd(tt, m, measures, values, 1/10**hexp)
                       )**2
            print(f"1e-{hexp:d}: {np.sqrt(err):.2e}")
        print()

    ProjectionTest = True
    if ProjectionTest:
        print("Tangent test:")
        print("-------------")

        num_samples = 20000              # number of samples used for the Monte-Carlo integration
        dimensions = [3]*4               # dimensions of the coefficient tensor
        ranks = [3]*(len(dimensions)-1)  # TT-ranks of the coefficient tensor
        print(f"Number of samples: {num_samples}")
        print(f"Dimensions:        {dimensions}")
        print(f"Ranks:             {ranks}")
        print()

        # Define a random initial value.
        tt = xe.TTTensor.random(dimensions, ranks)
        componentGradients = []
        gradient = xe.TTTensor(tt.dimensions)
        tt.move_core(0)
        for m in range(tt.order()):
            g = xe.TTTensor(tt)
            pg = xe.TTTensor(tt)
            assert g.corePosition == m
            core = np.random.randn(*tt.get_component(m).dimensions)
            g.set_component(m, xe.Tensor.from_buffer(core))
            componentGradients.append(g)
            if m < tt.order()-1:
                tt.move_core(m+1)
            U = tt.get_component(m).to_ndarray()
            core -= np.einsum('lex, yzx, yzr -> ler', U, U, core)
            pg.set_component(m, xe.Tensor.from_buffer(core))
            gradient = gradient + pg

        ts = TangentSpace(tt)
        # tangentVector = project_component_gradients(tt, componentGradients)  #TODO: ... (should,eg, both be equal?)
        tangentVector = ts.project(gradient)


        print( "Consider a tangent space at tt and let M = tt.order() and tt.corePosition == M-1.")
        print( "(1)  project(tt)[m]   == 0   (for m < M-1)")
        print( "     project(tt)[M-1] == tt.component[M-1]")
        tt.move_core(tt.order()-1)
        tmp = ts.project(tt)
        for m in range(tt.order()-1):
            print(f"Error[{m+1}/{tt.order()}]: {np.linalg.norm(tmp.dX[m])/xe.frob_norm(tt):.0e}")
        m = tt.order()-1
        print(f"Error[{m+1}/{tt.order()}]: {np.linalg.norm(tmp.dX[m] - tt.get_component(m).to_ndarray())/xe.frob_norm(tt):.0e}\n")

        print( "(2)  TTTensor(project(tt)) == tt")
        error = ts.project(tt).to_TTTensor() - tt
        print(f"Error: {xe.frob_norm(error)/xe.frob_norm(tt):.0e}\n")

        print( "(3)  project(TTTensor(tangentVector)) == tangentVector")
        error = ts.project(tangentVector.to_TTTensor()) - tangentVector
        print(f"Error: {error.norm()/tangentVector.norm():.0e}\n")

        print( "(4)  project(project(gradient)) == project(gradien)")
        error = ts.project(tangentVector) - tangentVector
        print(f"Error: {error.norm()/tangentVector.norm():.0e}\n")

        def coreProjection(_tt, _k):
            assert 0 <= _k < _tt.order()
            assert _tt.dimensions == tt.dimensions
            ret = xe.TTTensor(tt)
            ret.move_core(_k)

            leftContraction = np.ones((1,1))
            for pos in range(_k):
                leftContraction = np.einsum('lk, ler, kes -> rs', leftContraction, ret.get_component(pos).to_ndarray(), _tt.get_component(pos).to_ndarray())

            rightContraction = np.ones((1,1))
            for pos in reversed(range(_k+1,tt.order())):
                rightContraction = np.einsum('ler, kes, rs -> lk', ret.get_component(pos).to_ndarray(), _tt.get_component(pos).to_ndarray(), rightContraction)

            core = np.einsum('lk, kes, rs -> ler', leftContraction, _tt.get_component(_k).to_ndarray(), rightContraction)
            ret.set_component(_k, xe.Tensor.from_buffer(core))
            return ret

        #TODO: Du solltest NIE einzelne Cores speichern. Denn die sind nicht eindeutig (gauge condition...)

        print( "Assume that all but one component gradients are zero.")
        print( "Then the gradient can be represented explicitly by replacing the core of tt.")
        print( "(5)  TTTensor(project(componentGradient)) == componentGradient")
        for m in range(tt.order()):
            tmp = ts.project(componentGradients[m]).to_TTTensor()
            print(f"Error[{m+1}/{tt.order()}]: {xe.frob_norm(componentGradients[m]-tmp)/xe.frob_norm(componentGradients[m]):.0e}")
        print()

        print( "(6)  coreProjection(componentGradient) == componentGradient")
        for m in range(tt.order()):
            tmp = coreProjection(componentGradients[m], m)
            print(f"Error[{m+1}/{tt.order()}]: {xe.frob_norm(componentGradients[m]-tmp)/xe.frob_norm(componentGradients[m]):.0e}")
        print()

        # print( "(7)  coreProjection(tangentVector) == componentGradient")
        # for m in range(tt.order()):
        #     tmp1 = coreProjection(componentGradients[m], m)
        #     tmp2 = coreProjection(gradient, m)
        #     # tmp2 = coreProjection(tangentVector.to_TTTensor(), m)
        #     #TODO: Das muss nicht sein. tangentVector ist eine Summe von componentGradient's.
        #     #      Die coreProjection von dieser Summe ist nur dann wieder der componentGradient,
        #     #      wenn alle anderen Summanden orthogonal auf dem m-ten Tangentialraum stehen.
        #     #      Wenn das so wäre, dann gäbe es auch keinen Unterschied zwischen ASD und Riemannian SD.
        #     print(f"Error[{m+1}/{tt.order()}]: {xe.frob_norm(tmp1 - tmp2)/tangentVector.norm():.0e}")
        # print()
