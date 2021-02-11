#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:58:33 2020

@author: sallandt

This code runs the Policy Iteration algorithm using Tensor Trains as function
approximator.
Loads functions from the general 
"""

import xerus as xe
import numpy as np
from scipy import linalg as la
import valuefunction_TT, ode, pol_it
import time

run_set_dynamics = False
if run_set_dynamics == True:
    import set_dynamics
def main_p(nos = None):    
    vfun = valuefunction_TT.Valuefunction_TT()
    # vfun.test()
    testOde = ode.Ode()
    horizon = 1
    n_sweep = 1000
    rel_val_tol = 1e-6
    rel_tol = 1e-2
    max_pol_iter = 100
    max_iter_Phi = 1
    seed = 1
    
    dof = vfun.calc_dof()
    dof_factor = 6  # samples = dof_factor*(ordersoffreedom)
    if nos is None:
        nos = 1000000
    nos_test_set = int(nos/5)  # number of samples of the test set
    # print('number of samples', nos)
    
    polit_params = [nos, nos_test_set, n_sweep, rel_val_tol, rel_tol, max_pol_iter, max_iter_Phi, horizon, seed]
    
    # testOde.test()
    testpolit = pol_it.Pol_it(vfun, testOde, polit_params)
    
    
    t00 = time.time()
    t01 = time.perf_counter()
    # solve HJB
    testpolit.solve_HJB()
    t10 = time.time()
    t11 = time.perf_counter()
    print('The calculations took:, time(), perf_counter()', t10 - t00, t11 - t01 )
    # xe.save_to_file(testpolit.v.V, 'V_optimized')
