
import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy
import matplotlib.pyplot as plt
import pickle
from misc import compute_discounted_payoff_PutAmer

class Pol_it:
    def __init__(self, initial_valuefun, ode, polit_params):
        self.v = initial_valuefun
        self.ode = ode
        self.v.set_add_fun(self.ode.calc_reward)
        [self.nos, self.nos_test_set, self.n_sweep, self.rel_val_tol, self.rel_tol, self.max_pol_iter, self.max_iter_Phi, self.horizon, self.seed] = polit_params
        load_me = np.load('save_me.npy')
        self.interval_half = load_me[2]
        self.t_vec_p = self.v.t_vec_p
        self.t_vec_s = self.ode.t_vec_s
        self.current_time = 0
        self.current_end_time = 0
        self.curr_ind = 0
        self.x0 = np.load('x0.npy')
        self.samples, self.samples_test = self.build_samples(-self.interval_half, self.interval_half)
        self.data_x = self.v.prepare_data_before_opt(self.samples)
        self.constraints_list = self.construct_constraints_list()
        self.c = self.ode.calc_reward(0, self.samples[:,:,-1])
        self.c_test = self.ode.calc_reward(0, self.samples_test[:,:,-1])
        self.update = True



    def build_samples(self, samples_min, samples_max):
        samples_dim = self.ode.n
        samples_mat = np.zeros(shape=(samples_dim, self.nos, len(self.t_vec_p)))
        samples_mat_test = np.zeros(shape=(samples_dim, self.nos_test_set, len(self.t_vec_p)))




        load_me = np.load('save_me.npy')
        interest = load_me[7]
        strike_price = load_me[8]
        trend = load_me[9]*np.ones(samples_dim)
        maturity = self.v.t_vec_p[-1]
        spot = self.x0
        correlation = load_me[10]
        sigma = load_me[6]*np.ones(samples_dim)
        

        # print('interest', interest, 'strike_price', strike_price, 'trend', trend, 'maturity', maturity, 'spot', spot, 'correlation', correlation, 'sigma', sigma)

        _, samples_mat, disc_payoff = compute_discounted_payoff_PutAmer(samples_dim, samples_mat.shape[2]-1, samples_mat.shape[1], spot, strike_price, maturity, trend, sigma, correlation, interest, self.seed)
        samples_mat = np.transpose(samples_mat, (1,0,2))
        _, samples_mat_test, __ = compute_discounted_payoff_PutAmer(samples_dim, samples_mat_test.shape[2]-1, samples_mat.shape[1], spot, strike_price, maturity, trend, sigma, correlation, interest, self.seed+1)
        samples_mat_test = np.transpose(samples_mat_test, (1,0,2))

#         samples_dim = self.ode.n
#         samples_mat = np.zeros(shape=(samples_dim, self.nos, len(self.t_vec_p)))
#         np.random.seed(self.seed)
#         for i0 in range(self.nos):
#             samples_mat[:, i0, 0] = self.x0
#         for i0 in range(self.nos_test_set):
#             samples_mat_test[:, i0, 0] = self.x0
# 
#         step_per = int(np.round(self.v.tau / self.ode.tau))
#         # print('step_per', step_per)
#         for i0 in range(samples_mat.shape[2] - 1):
#             curr = samples_mat[:, :, i0]
#             # curr = 1*self.ode.step(self.t_vec_s[i0], samples_mat[:, :, i0], None)
#             for i1 in range(step_per):
#                 curr = self.ode.step(self.t_vec_s[i0], curr, None)
#             samples_mat[:, :, i0+1] = curr
#         for i0 in range(samples_mat_test.shape[2] - 1):
#             curr = samples_mat_test[:, :, i0]
#             # curr = 1*self.ode.step(self.t_vec_s[i0], samples_mat[:, :, i0], None)
#             for i1 in range(step_per):
#                 curr = self.ode.step(self.t_vec_s[i0], curr, None)
#             samples_mat_test[:, :, i0+1] = curr
        # print('min, max', np.min(samples_mat), np.max(samples_mat))
        # print('min, max', np.min(samples_mat_test), np.max(samples_mat_test))
#         print('samples_mat.shape', samples_mat.shape)
#         mean = np.mean(samples_mat[:,:,-1], axis=0)
#         print('mean.shape', mean.shape)
#         val = np.maximum(0, 100 - mean)
#         print('val.shape', val.shape)
#         avg = np.exp(-0.1*self.ode.t_vec_s[-1]) * np.mean(val)
#         print('euro_philipp', np.mean(disc_payoff[:,-1]))
#         print('value as european option:', avg)
#         input()

        return samples_mat, samples_mat_test





    def construct_constraints_list(self):
        # return None
        n = self.ode.n
        xvec = np.zeros(shape=(n, n+1))
        P_list = self.v.P_batch(xvec)
        dP_list = self.v.dP_batch(xvec)
        for i0 in range(n):
            P_list[i0][:,i0] = dP_list[i0][:,i0]
        return P_list

    
    def solve_HJB(self, start_num = None):
        if type(start_num) is not int:
            start_num = len(self.t_vec_p) -2
        pickle.dump(self.v.V[len(self.t_vec_p) -1], open('V_{}'.format(str(len(self.t_vec_p) -1)), 'wb'))
        
        for i0 in range(start_num, -1, -1):
            self.curr_ind = i0
            if i0 is not start_num:
                # print('set V', i0)
                self.v.V[self.curr_ind] = self.v.V[self.curr_ind+1]
            self.current_time = self.v.t_vec_p[i0]
            ind_end = np.minimum(len(self.t_vec_p) - 1, i0+self.horizon)
            self.current_end_time = self.v.t_vec_p[ind_end]
            # print('ind_end', ind_end, 't_start, t_end', self.current_time, self.current_end_time)
            self.solve_HJB_fixed_time()


    def solve_HJB_fixed_time(self):
        curr_samples, rew_MC, curr_payout, mask = self.build_rhs_batch()
        curr_samples_test, rew_MC_test, curr_payout_test, mask_test = self.build_rhs_batch(True)
        # print('shapes after', curr_samples.shape, rew_MC.shape)
        # rew_MC_test = self.build_rhs_batch(self.samples_test)
        if self.current_time == 0:
            avg = 1/self.samples.shape[1]*np.sum((self.c*np.exp(-self.ode.interest*(self.current_end_time - self.current_time))))
            avg_test = 1/self.samples_test.shape[1]*np.sum((self.c_test*np.exp(-self.ode.interest*(self.current_end_time - self.current_time))))
            curr_samples = self.samples[:,:,0] + np.random.randn(self.samples[:,:,0].shape[0], self.samples[:,:,0].shape[1])
            curr_samples_test = self.samples_test[:,:,0] + np.random.randn(self.samples_test[:,:,0].shape[0], self.samples_test[:,:,0].shape[1])
            rew_MC = np.ones(curr_samples.shape[1])*avg
            rew_MC_test = np.ones(curr_samples_test.shape[1])*avg_test
            self.v.V[0].round([1]*len(self.v.V[0].ranks()))

        # print('rewmc', rew_MC)
        data_y = self.v.prepare_data_while_opt(self.current_time, curr_samples)

        data = [self.data_x, data_y[0], data_y[1], data_y[2], rew_MC, self.constraints_list, self.calc_mean_error, curr_samples_test, rew_MC_test]
        params = [self.n_sweep, self.rel_val_tol]
        # print('rhs built')
        
        if self.update:
            self.v.solve_linear_HJB(data, params)
        # postprocessing
        curr_larger = (curr_payout >= self.v.eval_V(self.current_time, self.samples[:, :, self.curr_ind]))
        curr_larger_test = (curr_payout_test >= self.v.eval_V(self.current_time, self.samples_test[:, :, self.curr_ind]))
        # print('self.curr_ind', self.curr_ind)
        # print('curr_larger', curr_larger)
        # print('mask', mask)
        # print('curr_payout', curr_payout)
        # print('evalv',  self.v.eval_V(self.current_time, curr_samples))
        # print('rew_MC',  rew_MC)
        mask_gv = np.all( [curr_larger,  mask], axis=0)
        mask_gv_test = np.all( [curr_larger_test,  mask_test], axis=0)
        # print('mask_gv', mask_gv)
        # print('self.c before', self.c)
        if self.current_time != 0:
            self.c[mask_gv] = curr_payout[mask_gv]
            self.c[~mask_gv] = (self.c[~mask_gv]*np.exp(-self.ode.interest*(self.current_end_time - self.current_time)))
            self.c_test[mask_gv_test] = curr_payout_test[mask_gv_test]
            self.c_test[~mask_gv_test] = (self.c_test[~mask_gv_test]*np.exp(-self.ode.interest*(self.current_end_time - self.current_time)))
        else:
            self.c = self.c*np.exp(-self.ode.interest*(self.current_end_time - self.current_time))
            self.c_test = self.c_test*np.exp(-self.ode.interest*(self.current_end_time - self.current_time))
            print('price', 1/self.samples.shape[1]*np.sum(self.c))
            print('price_test', 1/self.samples_test.shape[1]*np.sum(self.c_test))

        # print('self.c after update', self.c, 'discount', np.exp(-self.ode.interest*(self.current_end_time - self.current_time)), 'curr and end time', self.current_end_time , self.current_time)
        # print('mask_gv', mask_gv)

        # plt.figure()
        # x = np.arange(100)
        # plt.plot(x, self.v.eval_V(self.current_time, x.reshape((1, x.size))), c='r')
        # plt.scatter(curr_samples.reshape((rew_MC.size)), rew_MC)
        # plt.scatter(curr_samples.reshape((rew_MC.size)), self.v.eval_V(self.current_time, curr_samples), c='r')
        # plt.figure()
        # plt.scatter(self.samples[:, :, self.curr_ind].reshape((curr_payout.size)), curr_payout)
        # plt.scatter(self.samples[:, :, self.curr_ind].reshape((curr_payout.size)), self.v.eval_V(self.current_time, self.samples[:, :, self.curr_ind]), c='r')
        # plt.plot(x, self.v.eval_V(self.current_time, x.reshape((1, x.size))), c='r')
        # plt.show()
        # input()
        pickle.dump(self.v.V[self.curr_ind], open('V_{}'.format(str(self.curr_ind)), 'wb'))
        np.save('c_add_fun_list', self.v.c_add_fun_list)
        try:
            rel_diff = xe.frob_norm(self.v.V[self.curr_ind] - V_old) / xe.frob_norm(V_old)
        except:
            rel_diff = 1
        # mean_error_test_set = self.calc_mean_error(self.samples_test, y_mat_test, rew_MC_test)
        # print('frob_norm(V)', xe.frob_norm(self.v.V[self.curr_ind]))
    
    def calc_mean_error(self, V, xmat, rew_MC):
        error = (self.v.eval_V(V, xmat) - rew_MC)
        return np.linalg.norm(error)**2 / rew_MC.size


    def calc_u(self, t, x):
        grad = self.v.calc_grad(t, x)
        return self.ode.calc_u(t, x, grad)

    
    def build_rhs_batch(self, validation = False):
        points = np.linspace(self.v.t_to_ind(self.current_time), self.v.t_to_ind(self.current_end_time), 2)
        t_points = np.linspace(self.current_time, self.current_end_time, int((self.current_end_time - self.current_time)/self.v.tau)+1  )

        if not validation:
            curr_samples = self.samples[:,:,int(np.round(points[0]))]
            curr_payout = self.ode.calc_reward(self.current_time, curr_samples)
            mask = curr_payout > 0
            rew_MC = np.exp(-self.ode.interest*(self.current_end_time - self.current_time))*self.c
        else:
            curr_samples = self.samples_test[:,:,int(np.round(points[0]))]
            curr_payout = self.ode.calc_reward(self.current_time, curr_samples)
            mask = curr_payout > 0
            rew_MC = np.exp(-self.ode.interest*(self.current_end_time - self.current_time))*self.c_test

        rew_MC = rew_MC[mask]
        samples = curr_samples[:, mask]

        return samples, rew_MC, curr_payout, mask
        




