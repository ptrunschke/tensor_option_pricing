#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:02:03 2019

@author: sallandt

Builds system matrices and saves them. Also calculates an initial control. xerus dependencies can be deleted.
"""
import xerus as xe
import numpy as np
from scipy import linalg as la
import pickle

def set_dynamics(pol_deg = None, num_valuefunctions = None, n=None):
    # num_valuefunctions = 3
    # num_valuefunctions = 6
    # num_valuefunctions = 11
    # num_valuefunctions = 21
    if n is None:
        n = 5
    if pol_deg is None:
        pol_deg = 2
    if num_valuefunctions is None:
        num_valuefunctions = 4

    T = 3


    sigma = .2
    interest = 0.05
    increase = 0.1
    strike_price = 100
    interval_min = 0 # integration area of HJB equation is [interval_min, interval_max]**n
    interval_max = 500
    x0 = np.ones(n)*100
    rho = 0

    # print('x0', x0)
    rank = 3
    payoff_vec = 1/n*np.ones(n)
    # print('payoff_vec', payoff_vec)
    
    
    Gamma_mat = rho*np.ones((n,n))
    np.fill_diagonal(Gamma_mat, 1)
    # print('gammamat/covariance', Gamma_mat)
    L = la.sqrtm(Gamma_mat)
    # print('L', L)
    
    t_vec_p = np.linspace(0, T, num_valuefunctions)
    # tau = 1e-2 # time step size
    tau = t_vec_p[1] - t_vec_p[0] # time step size
    num_timesteps = int(np.round((T - 0) / tau))
    t_vec_s = np.linspace(0, T, num_timesteps+1)
    # print('t_vec_p', t_vec_p)
    # print('t_vec_s', t_vec_s)
    
    b = 1 # left end of Domain
    a = -1 # right end of Domain
    nu = 1 # diffusion constant
    lambd = 0.1 # cost parameter
    gamma = 0 # discount factor, 0 for no discount
    boundary = 'Neumann' # use 'Neumann' or "Dirichlet
    use_full_model = True # if False, model is reduced to r Dimensions
    r = n # Model is reduced to r dimensions, only if use_full_model == False
    load = np.zeros([11])
    load[0] = lambd; load[1] = gamma; load[2] = interval_min; load[3] = interval_max; load[4] = tau; load[5] = n; load[6] = sigma; load[7] = interest; load[8] = strike_price; load[9] = increase; load[10] = rho
    
    np.save("x0", x0)
    np.save('payoff_vec.npy', payoff_vec)
    np.save('L', L)
    np.save("save_me", load)
    np.save('t_vec_p', t_vec_p)
    np.save('t_vec_s', t_vec_s)
    #
    
    'delete from here if you do not want to use xerus'
    
    set_V_new = True
    print(set_V_new)
    
    import orth_pol
    
    load_me = np.load('save_me.npy')
    pol, dpol = orth_pol.calc_pol(interval_max, interval_min, 2)
    
    desired_ranks = [rank]*(n-1)
    # V_setranks = xe.TTTensor.random([pol_deg+1]*n, desired_ranks)
    # desired_ranks = V_setranks.ranks()
    load_prev = False
    if load_prev == True:
        # V_prev = xe.load_from_file('V_optimized.txt')  # load as initial guess
        V_prev = xe.load_from_file('V_99.txt')  # load as initial guess
        if V_prev.order() < r:
            print('increase dimensions')
            print((V_prev.ranks() + [1]))
            V_increased = xe.TTTensor.random(V_prev.dimensions + [pol_deg], V_prev.ranks() + [1])
            for i0 in range(V_prev.order()):
                V_increased.set_component(i0, V_prev.get_component(i0))
            V_increased.set_component(V_prev.order(), xe.Tensor.dirac((1,pol_deg,1), [0,0,0]))
            V_prev = V_increased
    else:
        V_prev = 0
    
    if type(V_prev) == xe.TTTensor:
        print("type(V_prev) == xe.TTTensor")
        V_new = V_prev
        new = pol_deg+1
        prev = V_prev.dimensions
        print("prev dim:", prev) 
        for iter_0 in range(r):
            comp = V_prev.get_component(iter_0)
            print(comp.dimensions)
            comp.resize_mode(mode=1, newDim=new, cutPos=prev[iter_0])
            if iter_0 != 0:
                if comp.dimensions[0] != desired_ranks[iter_0-1]:
                    comp.resize_mode(mode=0, newDim=desired_ranks[iter_0-1], cutPos=comp.dimensions[0])
                    comp = comp + 0.000000000001*xe.Tensor.random(comp.dimensions)
            if iter_0 != r-1:
                if comp.dimensions[2] != desired_ranks[iter_0]:
                    comp.resize_mode(mode=2, newDim=desired_ranks[iter_0], cutPos=comp.dimensions[2])
                    comp = comp + 0.000000000001*xe.Tensor.random(comp.dimensions)
            V_new.set_component(iter_0, comp)
        V_new.canonicalize_left()
    else:
        # V_new = 0.000001*xe.TTTensor.random([pol_deg + 1]*n, desired_ranks)
        V_new = 0*xe.TTTensor.dirac([pol_deg + 1]*n, [0]*(n))
    # V_new = V_new / xe.frob_norm(V_new)
    print('norm V_new', xe.frob_norm(V_new))
    pickle.dump(V_new, open("V_new", 'wb'))
