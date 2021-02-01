"""
This file implements the following functions:
    armijo_step():                         Computes the optimal step size according to the Armijo rule.
The classes TangentSpace and TangentVector are defined to encapsulate all computations concerning tangent vectors of a TT-Manifold.
"""
import copy

import numpy as np
import xerus as xe


################################################################################
#    Classes                                                                   #
################################################################################

class TangentSpace(object):
    def __init__(self, _tt):
        self._U = xe.TTTensor(_tt)
        self._U.move_core(_tt.order()-1)
        self._V = xe.TTTensor(_tt)
        self._V.move_core(0)

    @property
    def dimensions(self): return self._U.dimensions

    @property
    def ranks(self): return self._U.ranks()

    @property
    def order(self): return self._U.order()

    def U(self, _mode):
        assert 0 <= _mode < self.order-1
        return self._U.get_component(_mode).to_ndarray()

    def V(self, _mode):
        assert 0 < _mode <= self.order-1
        return self._V.get_component(_mode).to_ndarray()

    def project(self, _obj):
        if isinstance(_obj, xe.TTTensor):
            assert _obj.dimensions == self.dimensions
            rightStack = [np.ones((1,1))]
            for pos in reversed(range(1,self.order)):
                rightStack.append(np.einsum('ler, kes, rs -> lk', self.V(pos), _obj.get_component(pos).to_ndarray(), rightStack[-1]))
            leftStackTop = np.ones((1,1))
            dX = []
            for pos in range(self.order):
                dX.append(np.einsum('lk,kes,rs -> ler', leftStackTop, _obj.get_component(pos).to_ndarray(), rightStack[-1]))
                if pos < self.order-1:
                    leftStackTop = np.einsum('lk, ler, kes -> rs', leftStackTop, self.U(pos), _obj.get_component(pos).to_ndarray())
                    rightStack.pop()
            for pos in range(self.order-1):
                proj = np.einsum('lex, yzx, yzr -> ler', self.U(pos), self.U(pos), dX[pos])
                dX[pos] -= proj
            ret = TangentVector(self)
            ret._dX = dX
            return ret
        elif isinstance(_obj, TangentVector):
            return self.project(_obj.to_TTTensor())  #TODO: make faster (to_TTTensor is not necessary)
        else:
            raise TypeError("...")


class TangentVector(object):
    def __init__(self, _space):
        if isinstance(_space, xe.TTTensor):
            tt = _space
            self._space = TangentSpace(tt)
            self._dX = [np.zeros(tt.get_component(m).dimensions) for m in range(self.order-1)] \
                     + [self.space._U.get_component(self.order-1).to_ndarray()]
        elif isinstance(_space, TangentVector):
            tv = _space
            self._space = tv._space
            self._dX = [dXc.copy() for dXc in tv._dX]
        elif isinstance(_space, TangentSpace):
            self._space = _space
            ranks = [1] + _space.ranks + [1]
            self._dX = [np.zeros((ranks[m], self.dimensions[m], ranks[m+1])) for m in range(self.order)]
        else:
            raise TypeError("...")

    @property
    def space(self): return self._space

    @property
    def dX(self): return self._dX

    @property
    def dimensions(self): return self.space.dimensions

    @property
    def ranks(self): return self.space.ranks

    @property
    def order(self): return self.space.order

    def copy(self): return TangentVector(self)

    def __matmul__(self, _other):
        assert isinstance(_other, TangentVector) and xe.frob_norm(self.space._U - _other.space._U) < 1e-8 + 1e-8*xe.frob_norm(self.space._U)
        return sum(np.einsum('ler,ler', selfCmp, otherCmp) for selfCmp, otherCmp in zip(self.dX, _other.dX))

    def norm(self):
        return np.sqrt(self @ self)

    def __mul__(self, _other):
        assert isinstance(_other, (float, int))
        _other = np.float_(_other)
        ret = self.copy()
        for m in range(self.order):
            ret.dX[m] *= _other
        return ret

    def __rmul__(self, _other):
        return self * _other

    def __div__(self, _other):
        return self * (1/_other)

    def __add__(self, _other):
        assert isinstance(_other, TangentVector) and xe.frob_norm(self.space._U - _other.space._U) < 1e-8 + 1e-8*xe.frob_norm(self.space._U)
        ret = self.copy()
        for m in range(self.order):
            ret.dX[m] += _other.dX[m]
        return ret

    def __neg__(self):
        return (-1)*self

    def __sub__(self, _other):
        return self + (-_other)

    def to_TTTensor(self):
        # This is the tangent vector T on TT manifolds at point X:
        # Each tangent vector can be written as
        #     T = dX[0]*V[1]*...*V[d-1] + U[0]*dX[1]*V[2]*...*V[d-1] + ... + U[0]*...*U[d-2]*dX[d-1]
        # where the dX[k] are arbitrary component tensors and the U[j] and V[j] are the complenent tensors of the left and right canonicalized TT tensor X.
        # Note that the k-th summand is just X with the core at position k and the the k-th component replaced by dX[k].
        # The rank of these tensors is bounded by 2*rank(X).
        # This can be seen by the following construction:
        #     T = [[U[0], dX[0]]] * [[U[1], dX[1]], [0, V[1]]] * ... * [[dX[d-1]], [V[d-1]]]
        # Using this construction we can write this as:
        ret = xe.TTTensor(self.dimensions)
        if self.order == 1:
            # ret = xe.TTTensor([self.dX[0].shape[1]])
            ret.set_component(0, xe.Tensor.from_buffer(self.dX[0]))
            return ret
        Um = np.asarray(self.space.U(0))
        dXm = self.dX[0]
        ret.set_component(0, xe.Tensor.from_buffer(np.block([[[dXm, Um]]])))
        for m in range(1, self.order-1):
            Um = np.asarray(self.space.U(m))
            Vm = np.asarray(self.space.V(m))
            dXm = self.dX[m]
            Zm = np.zeros((Vm.shape[0], Um.shape[1], Um.shape[2]))
            ret.set_component(m, xe.Tensor.from_buffer(np.block([[[Vm, Zm]], [[dXm, Um]]])))
        Vm = np.asarray(self.space.V(self.order-1))
        dXm = self.dX[self.order-1]
        ret.set_component(self.order-1, xe.Tensor.from_buffer(np.block([[[Vm]], [[dXm]]])))
        ret.canonicalize_left()
        return ret


################################################################################
#    Functions                                                                 #
################################################################################

def project_component_gradients(_tt, _componentGradients):
    def tensor(_obj):
        return xe.Tensor.from_buffer(_obj)
    ts = TangentSpace(_tt)
    return sum((ts.project(cp) for cp in _componentGradients), start=TangentVector(ts))


def armijo_step(_retraction, _costs, _descentDirGrad, _maxIterations=200, _initialStepSize=1, _beta=0.8, _gamma=1e-4):
    """
    Compute the optimal step size according to the Armijo rule.

    The algorithm starts with a step size of 1. In each iteration it checks if
    the costs for the current step deceed the initial costs by a margin
    proportional to `_gamma` and `_gradientNorm**2`. If this is not the case it
    reduces the step size by a factor of `_beta`.


    Parameters
    ----------
    _retraction : callable
        A function that takes a step size and returning a TTTensor.
    _costs : callable
        A function that rakes a TTTensor and returns its costs.
    _gradientNorm : float
        The norm of the gradient that is used in `_retraction`.
    _maxIterations: int
    _beta, _gamma : float
        Parameters for the Armijo rule.

    Returns
    -------
    TTTensor
        The tensor for the optimal step size.
    """
    #TODO:
    # wee also need 0 < c1 < c2 < 0.5 s.t.
    #     if trialCosts <= initialCosts - c1*stepSize*_descentDirGrad
    # AND a condition on the new gradient
    assert _descentDirGrad > 0

    def update(_state):
        _state['x'], _state['retractionError'] = _retraction(_state['stepSize'])
        _state['costs'] = _costs(_state['x'])

    zeroState = {'stepSize': 0.0}
    update(zeroState)
    initialState = {'stepSize': max(_initialStepSize, 1e-6)}
    update(initialState)

    print("Armijo UP")
    state = copy.deepcopy(initialState)
    bestState = copy.deepcopy(state)
    for iteration in range(_maxIterations):
        state['stepSize'] /= _beta
        update(state)
        if state['costs'] < bestState['costs']:
            bestState = copy.deepcopy(state)
        print(f"Armijo[{state['stepSize']:.2e}]  Current costs: {state['costs']:.4e}  |  Best costs: {bestState['costs']:.4e}  |  Costs for step size 0: {zeroState['costs']:.4e}")
        if state['costs'] <= zeroState['costs'] - state['stepSize']*_descentDirGrad:
            break
        elif state['costs'] > initialState['costs']:
            break
    bestUpState = bestState

    print("Armijo DOWN")
    state = copy.deepcopy(initialState)
    bestState = copy.deepcopy(state)
    for iteration in reversed(range(_maxIterations - iteration)):
        state['stepSize'] *= _beta
        update(state)
        if state['costs'] < bestState['costs']:
            bestState = copy.deepcopy(state)
        print(f"Armijo[{state['stepSize']:.2e}]  Current costs: {state['costs']:.4e}  |  Best costs: {bestState['costs']:.4e}  |  Costs for step size 0: {zeroState['costs']:.4e}")
        if state['costs'] <= zeroState['costs'] - state['stepSize']*_descentDirGrad:
            break
        elif state['costs'] > initialState['costs']:
            break
        elif state['costs'] > bestState['costs'] and bestState['costs'] < initialState['costs']:  # The step size is too small and increases the costs again.
            break

    if iteration == 0:
        print("WARNING: _maxIterations exceeded")

    if bestUpState['costs'] < bestState['costs']:
        bestState = bestUpState

    return bestState['x'], bestState['retractionError'], bestState['stepSize']

    # _initialStepSize = max(_initialStepSize, 1e-6)

    # initial, initialRetractionError = _retraction(0)
    # initialCosts = _costs(initial)

    # stepSize = _initialStepSize
    # trial, trialRetractionError = _retraction(stepSize)
    # trialCosts = _costs(trial)

    # best = trial
    # bestCosts = trialCosts
    # bestRetractionError = trialRetractionError
    # bestStepSize = stepSize

    # zeroCosts = initialCosts
    # initialCosts = None
    # for itr in range(_maxIterations):
    #     trial, trialRetractionError = _retraction(stepSize)
    #     trialCosts = _costs(trial)
    #     if initialCosts is None:
    #         initialCosts = trialCosts
    #     if trialCosts < bestCosts:    # the new step size decreases the costs further
    #         best = trial
    #         bestCosts = trialCosts
    #         bestRetractionError = trialRetractionError
    #         bestStepSize = stepSize
    #     print(f"Armijo[{stepSize:.2e}]  Current costs: {trialCosts:.4e}  |  Best costs: {bestCosts:.4e}  |  Initial costs: {initialCosts:.4e}")
    #     if trialCosts <= zeroCosts - stepSize*_descentDirGrad:
    #         print(f"Armijo: Terminating. trialCosts deceed zeroCosts")
    #         return best, bestRetractionError, bestStepSize
    #     elif trialCosts > bestCosts:  # the step size is too large now it increases the costs again
    #         break
    #     if itr < _maxIterations-1:
    #         stepSize /= _beta
    # _maxIterations -= itr
    # initialCosts = None
    # for itr in range(_maxIterations):
    #     trial, trialRetractionError = _retraction(stepSize)
    #     trialCosts = _costs(trial)
    #     if initialCosts is None:
    #         initialCosts = trialCosts
    #     if trialCosts < bestCosts:    # the new step size decreases the costs further
    #         best = trial
    #         bestCosts = trialCosts
    #         bestRetractionError = trialRetractionError
    #         bestStepSize = stepSize
    #     print(f"Armijo[{stepSize:.2e}]  Current costs: {trialCosts:.4e}  |  Best costs: {bestCosts:.4e}  |  Initial costs: {initialCosts:.4e}")
    #     if trialCosts <= zeroCosts - stepSize*_descentDirGrad:
    #         print(f"Armijo: Terminating. trialCosts deceed zeroCosts")
    #         return best, bestRetractionError, bestStepSize
    #     elif trialCosts > bestCosts and bestCosts < min(initialCosts, upCosts):  # the step size is too low now it increases the costs again
    #         print(f"Armijo: Terminating. trialCosts increases again")
    #         # Folgendes passiert: Die erste Schrittweite gibt schon einen Fehler < Fehler mit Schrittweite 0.
    #         #                     Aber die Schrittweite erhöhen vergrößert den Fehler nur.
    #         #                     Wenn er dann in diese Schleife geht, dann terminiert er sofort.
    #         #                     Was man also machen sollte: Beide schleifen logisch von einander trennen.
    #         #                     Entweder erhöht armijo die schrittweite oder er verringert sie. Er sollte sie nicht erst erhöhen und dann verringern.
    #         #                     Das sollte also eher sein:
    #         #                     bestUp, bestRetractionErrorUp, bestStepSizeUp = armijo_up(...)  # subroutine...
    #         #                     bestDown, bestRetractionErrorDown, bestStepSizeDown = armijo_down(...)  # subroutine...
    #         #                     return the one that minimizes the costs
    #         return best, bestRetractionError, bestStepSize
    #     if itr < _maxIterations-1:
    #         stepSize *= _beta
    # print("WARNING: _maxIterations exceeded")
    # return trial, trialRetractionError, stepSize
