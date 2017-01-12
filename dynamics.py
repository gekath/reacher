import numpy as np
from utils import *


class LinearGaussian():

    def __init__(self, K, k, covar, inv_covar):

        self.K = K
        self.k = k
        self.covar = covar
        self.inv_covar = inv_covar
        self.dimensions = K.shape


class Dynamics:

    def __init__(self):

        self.Fm = None
        self.fv = None
        self.covar = None
        self.x0sigma = None
        self.x0mu = None

    def fit(self, traj_state, traj_actions, reg=1):

        N, T, dx = traj_state.shape
        du = traj_actions.shape[2]

        self.Fm = np.zeros([T, dx, dx+du])
        self.fv = np.zeros([T, dx])
        self.covar = np.zeros([T, dx, dx])

        both_slice = slice(dx+du)
        xux_slice = slice(dx+du, dx+du+dx)

        for t in range(T - 1):
            xux = np.c_[traj_state[:,t,:], traj_actions[:,t,:], traj_state[:,t+1,:]]
            xux_mean = np.mean(xux, axis=0)
            xux_cov = (xux - xux_mean).T.dot(xux - xux_mean) / (N - 1)
            # xux_cov = np.cov(xux)

            sigma = 0.5 * (xux_cov +  xux_cov.T)
            sigma[both_slice, both_slice] += reg * np.eye(dx + du)
            # regularize ?

            Fm = np.linalg.pinv(sigma[both_slice, both_slice]).dot(sigma[both_slice, xux_slice]).T
            fv = xux_mean[xux_slice] - Fm.dot(xux_mean[both_slice])
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv

            covar = sigma[xux_slice, xux_slice] - Fm.dot(sigma[both_slice, both_slice]).dot(Fm.T)
            self.covar[t, :] = 0.5* (covar + covar.T)

            if t == 0:
                self.x0mu = np.mean(traj_state[:,t,:])
                self.x0sigma = np.diag(np.var(traj_state[:,t,:], axis=0))


class TrajectoryDistribution():

    def __init__(self, state, action, hyperparameters):

        self.dynamics = Dynamics()
        self.dynamics.fit(state, action)
        self.policy = self.backward(state, action, hyperparameters, self.dynamics)

    def backward(state, action, hyperparameters, dynamics):

        T, dx = state.shape
        _, du = action.shape

        fx = dynamics.fx
        fu = dynamics.fu

        K = np.zeros((T, du, dx))
        k = np.zeros((T,du))
        inv_covar = np.zeros((T,du, du))

        vt = np.zeros(dx)
        vtt = np.zeros((dx, dx))

        wu = hyperparameters['wu']
        wx = hyperparameters['wx']
        lamb = hyperparameters['lambda']

        cxx = compute_hessian(state, state, wx, wx)
        cux = compute_hessian(state, action, wx, wu)
        cuu = compute_hessian(action, action, wu, wu)

        cx = compute_jacobian(state, wx)
        cu = compute_jacobian(action, wu)

        # Backward pass
        for t in range(T-1, -1,-1):

            qx = cx[t,:] + np.dot(fx[t].T, vt)
            qu = cu[t,:] + np.dot(fu[t].T, vt)

            qxx = cxx[t,:,:] + np.dot(fx[t].T, np.dot(vtt, fx[t]))
            qux = cux[t,:,:] + np.dot(fu[t].T, np.dot(vtt, fx[t]))
            quu = cuu[t, :, :] + np.dot(fu[t].T, np.dot(vtt, fu[t]))

            # Use Levenberg - Marquardt heuristic to compute
            # inverse of quu
            eig_vals, eig_vecs = np.linalg.eig(quu)
            eig_vals[eig_vals < 0] = 0.0
            eig_vals += lamb
            quu_inv = np.dot(eig_vecs, np.dot(np.diag(1.0 / eig_vals), eig_vecs.T))
            inv_covar = quu_inv

            k[t] = -np.dot(quu_inv, qu)
            K[t] = -np.dot(quu_inv,qux)

            vt = qx - np.dot(K[t].T, np.dot(quu, k[t]))
            vtt = qxx - np.dot(K[t].T, np.dot(quu, K[t]))

        return LinearGaussian(K, k, inv_covar)

    def update(state, action):
        '''
        For each trajectory in given state/action pair, create a distribution,
         and return samples from this distribution.

        IN:
            state:      (T x dx) state vector
            actions:    (T x du) action vector

        OUT:

        '''
        # Construct a distribution for each trajectory.
        for traj in range(len(state)):

            dynamics = Dynamics()
            traj_state = state[traj,:]
            traj_action = action[traj,:]

            # TODO: get previous state, given trajectory, traj states, traj action
            # to get vals, acts

            T,dx = traj_state.shape
            _,du = traj_action.shape

            weights = { 'wx': [1/float(dx) for i in range(dx)],
                        'wu': [1/float(du) for i in range(du)]}

            # TODO:
            # dynamics.fit(vals, acts,.01)

            prev_mu, prev_sigma = forward(self.traj_dist, dynamics)

            for iter in range(_MAX_ITER):

                vals,acts



            for iter in range(_MAX_ITER):

                # Collect samples in simulation
                for sample in range(3):
                    s, a = get_sample(traj_dist,trajectory)
                    push_sample(trajectory, s, a)
                vals, acts = getPreviousSA(trajectory, traj_states, traj_actions)

                dynamics.fit(vals,acts,.01)

                traj_dist, new_eta = backward(traj_states, traj_actions, dynamics, eta, hyperparameters)
                print(new_eta)
                print('try again')
                mu, sigma = forward(traj_dist, dynamics)

                if new_eta > prev_eta:
                   min_eta = new_eta
                    # dynamics.fit(new_mu,new_sigma)

                # # TODO: calculate KL divergence between new traj_dist and prev_traj_dist
                # # check constraint, that kl_div <= _THRESHOLD
                # kl_div = calculate_KL_div(new_mu, prev_mu, traj_dist, prev_traj_dist)
                # if kl_div <= _THRESHOLD:
                #     break
                prev_traj_dist = traj_dist

            #Take initial sample
            samples = np.array([-np.random.multivariate_normal(mu[t], sigma[t], 1).flatten() for t in range(T)])
            commands = samples[:,9:]
            f = open('Traj1pred.txt', 'w')
            print('here')
            for act in commands:
                f.write("{}\n".format(" ".join([str(i) for i in act])))

            f.close()

            raw_input()
        return samples