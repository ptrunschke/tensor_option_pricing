import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy.integrate as integrate


class Ode:
    def __init__(self):
        load_me = np.load('save_me.npy')
        self.lambd = load_me[0]
        self.interval_min = load_me[2]
        self.interval_max = load_me[3]
        self.t_vec_s = np.load('t_vec_s.npy')
        self.tau = load_me[4]    
        self.n = int(load_me[5])
        self.sqrttau = np.sqrt(self.tau)
        self.sigma = load_me[6]
        self.interest = load_me[7]
        self.strike_price = load_me[8]
        self.increase = load_me[9]
        np.random.seed(2)
        self.L = np.load('L.npy')
        self.payoff_vec = np.load('payoff_vec.npy')

    def step(self, t, x, u, noise=None):
        return self.step_euler_mayurama(t, x, u, noise)
    
    def step_euler_mayurama(self, t, x, u, noise=None):
        if len(x.shape) == 1:
            if noise is not None:
                ret = x + self.tau*self.rhs_curr(t, x, u)  + (x*self.sigma*self.L @ noise)
            else:
                ret = x + self.tau*self.rhs_curr(t, x, u) + (x*self.sigma*self.L @ np.random.normal(loc=0.0,scale=self.sqrttau))
            # ret = np.minimum(ret, self.interval_max)
            # ret = np.maximum(ret, self.interval_min)
            return ret
        else:
            if noise is not None:
                ret = x+(self.tau*self.rhs_curr(t, x, u)  + x*(self.sigma*self.L @ noise))
            else:
                noise = x*(self.sigma*self.L @ np.random.normal(loc=0.0,scale=self.sqrttau,size=x.shape))
                ret = x + self.tau*self.rhs_curr(t, x, u) + noise
            # ret[ret > self.interval_max] = self.interval_max
            # ret[ret < self.interval_min] = self.interval_min
            return ret
        

    def rhs_curr(self, t, x, u):
        return self.f(t, x)


    def f(self, t, x):
        return (self.interest-self.increase) * x
        # return self.increase * x
        # return np.ones(x.shape)


    def solver(self, t_points, x, calc_u_fun):
        print('does not work for SDE')
        return 0


    def calc_reward(self, t, x, u=None):
        if len(x.shape) == 1:
            return self.g(x)
        else:
            return self.g_batch(x)

    def g_batch(self, x_mat):
        # return np.maximum(np.zeros(x_mat.shape[1]), strike_price - x_mat)
        # print('self.strike_price', self.strike_price, 'np.sum(x_mat, axis=0', np.sum(x_mat, axis=0), self.payoff_vec)
        return np.maximum(np.zeros(x_mat.shape[1]), -self.strike_price*np.ones(x_mat.shape[1]) + np.max(x_mat, axis=0))
        # return np.maximum(np.zeros(x_mat.shape[1]), self.strike_price*np.ones(x_mat.shape[1]) - np.sum(x_mat, axis=0)*self.payoff_vec[0])

    def g(self, x):
        return np.maximum(0, -self.strike_price + np.max(x))
    
    def calc_u(self, t, x, grad):
        if len(x.shape) == 1:
            return -self.g(t, x) * grad
        else:
            return -self.g(t, x) *grad


    def sample_boundary(self, no_dif_samples, samples_dim):
        samples_mat = np.zeros(shape=(samples_dim, no_dif_samples))
        for i0 in range(no_dif_samples):
            sample = np.random.uniform(-1, 1, samples_dim)
            sample /= np.linalg.norm(sample)/self.maxdiff
            samples_mat[:, i0] = self.x_target - sample
        return samples_mat

    def t_to_ind(self, t):
        # print('t, t/self.tau, int(t/self.tau', t, t/self.tau, int(t/self.tau))
        return int(t/self.tau)

    def test(self):
        t = 0.
        n = self.n
        m = self.n
        start = np.ones((n,2))
        print('start', start)
        control = np.zeros((m, 2))
        end = self.step(0, start, control)
        print('end', end)
        print('rewards', self.calc_reward(t, start, control), self.calc_reward(t, end, control))
        print('control', self.calc_u(t, start, start))
        return 0



# testOde = Ode()
# testOde.test()
