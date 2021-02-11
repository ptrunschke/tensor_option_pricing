import xerus as xe
import numpy as np
import set_dynamics, main_p, main_t

# pol_deg_vec = [2,3,4,5,6,7,8,9]
pol_deg_vec = np.arange(1, 8)
# pol_deg_vec = np.arange(1, 8)
# pol_deg_vec = [1, 2, 3, 4]
# num_samples_p = [5000000]
num_samples_p = [1000000]
# n_vec = [1000]
n_vec = [2,3,5,10,20,30,50,100,200,500,750,1000]
# n_vec = [750,1000]
# num_samples_p = [1000000, 5000000]
# num_samples_p = [20000, 200000, 1000000, 5000000]

# num_valuefunctions_vec = [21]
num_valuefunctions_vec = [9]
# num_valuefunctions_vec = [3, 6, 11]
# num_valuefunctions_vec = [11]
for i2 in num_samples_p:
    for i1 in num_valuefunctions_vec:
        for i3 in n_vec:
            for i0 in pol_deg_vec:
                print('pol_deg, num_valuefunctions, nos, n', i0, i1, i2, i3)
                # print('flip and sort active')
                set_dynamics.set_dynamics(i0, i1, i3)
                main_p.main_p(i2)
                main_t.main_t()
