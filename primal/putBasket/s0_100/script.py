import xerus as xe
import numpy as np
import set_dynamics, main_p, main_t

pol_deg_vec = [1,2,3,8,9]
# pol_deg_vec = np.arange(1, 9)
# pol_deg_vec = [2, 3,4]
# pol_deg_vec = [1, 2, 3, 4]
# num_samples_p = [5000000]
# num_samples_p = [1000000]
num_samples_p = [100000]
# num_samples_p = [20000, 200000, 1000000, 5000000]

# num_valuefunctions_vec = [6]
num_valuefunctions_vec = [31]
# num_valuefunctions_vec = [4, 9, 16]
# num_valuefunctions_vec = [4, 7]
# num_valuefunctions_vec = [3, 6, 11]
# num_valuefunctions_vec = [11]
for i2 in num_samples_p:
    for i1 in num_valuefunctions_vec:
        for i0 in pol_deg_vec:
            print('pol_deg, num_valuefunctions, nos', i0, i1, i2)
            set_dynamics.set_dynamics(i0, i1)
            main_p.main_p(i2)
            main_t.main_t()
