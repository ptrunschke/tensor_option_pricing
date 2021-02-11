#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:58:33 2020

@author: sallandt

"""

import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy
import scipy.stats
import valuefunction_TT, ode, pol_it
import matplotlib.pyplot as plt
import time

run_set_dynamics = False
if run_set_dynamics == True:
    import set_dynamics
    
def main_t():
    load_num = 'V_'
    
    vfun = valuefunction_TT.Valuefunction_TT(load_num, True, 'c_add_fun_list.npy')
    testOde = ode.Ode()
    vfun.set_add_fun(testOde.calc_reward)
    
    result_vec = []
    # testOde.test()
    load_me = np.load('save_me.npy')
    lambd = load_me[0]
    interval_min = load_me[2]
    interval_max = load_me[3]
    tau = load_me[4]    
    n = int(load_me[5])
    sigma = load_me[6]
    mu = load_me[7]
    
    num_samples = 100000
        
    for run in range(90, 100):
        seed = run
        polit_params = [num_samples, 2, 0, 0, 0, 0, 0, 0, seed]
        testpolit = pol_it.Pol_it(vfun, testOde, polit_params)
        not_stopped = np.ones(num_samples, dtype=bool)
        price_vec = np.zeros(num_samples)
        t_vec = vfun.t_vec_p
        # print('t_vec', t_vec)
        def eval_V(t, x):
            return vfun.eval_V(t, x)
        for i0 in range(len(vfun.V)-1):
            curr_eval = eval_V(t_vec[i0], testpolit.samples[:,:,i0])
            if i0 == 0:
                curr_eval[:] = 1e3
            # curr_eval = 5*np.ones(num_samples)
            curr_p = testOde.calc_reward(t_vec[i0], testpolit.samples[:,:,i0])
            curr_larger = curr_p >= curr_eval
            stop_now = np.all([curr_larger,  not_stopped], axis=0)
            not_stopped = np.all([~stop_now, not_stopped], axis=0)
            price_vec[stop_now] = curr_p[stop_now] * np.exp(-testOde.interest*(t_vec[i0] - t_vec[0]))
            # print('current time', t_vec[i0])
            # print('i0, curr_eval', i0, curr_eval)
            # print('i0, curr_p', i0, curr_p)
            # print('i0, curr_larger', i0, curr_larger)
            # print('i0, stop_now', i0, stop_now)
            # print('i0, not_stopped', i0, not_stopped)
            # print('i0, price_vec', i0, price_vec)
        curr_p = testOde.calc_reward(t_vec[-1], testpolit.samples[:,:,-1])
        price_vec[not_stopped] = curr_p[not_stopped] * np.exp(-testOde.interest*(t_vec[-1] - t_vec[0]))
        price = 1/num_samples*np.sum(price_vec)
        result_vec.append(price)
        # print('price', price)
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h
    
    mean, lower, upper = mean_confidence_interval(result_vec)
    print('------------ result -----------')
    # print(result_vec)
    print('mean', mean)
    # print('variance' , np.var(result_vec))
    print('std deviation', np.std(result_vec))
    
    # print('mean_confidence_interval (mean, lower, upper)', mean, lower, upper)
    print('confidence diff (minus, plus)', mean - lower, upper - mean)
    print('------------ done -----------')
    print(' ')
    print(' ')
    print(' ')
    
