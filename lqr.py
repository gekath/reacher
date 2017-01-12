import numpy as np
from dynamics import LinearGaussian, Dynamics
import scipy as sp
import scipy.linalg as splinalg
from function_names import *
# from Simulation import *

_MAX_ITER = 50
_THRESHOLD = 1.0

global iteration_state_repo
global iteration_action_repo
global state_norms
global action_norms

def set_norms(sn, sa):
    global state_norms
    global action_norms
    state_norms = sn
    action_norms = sa

def sample_dist(state, actions):

    global hyperparameters
    samples = []
    dynamics = Dynamics()
    # dynamics.fit(state,actions)
    for traj_no in range(state.shape[0]):
        # trajectory = 6

        dynamics = Dynamics()
        traj_states = state[traj_no,:,:]
        traj_actions = actions[traj_no,:,:]
        vals,acts = getPreviousSA(traj_no,traj_states, traj_actions)
        T, dx = traj_states.shape
        du = traj_actions.shape[1]

        hyperparameters = {'wx': [1/float(dx) for i in range(dx)],
                           'wu': [1/float(du) for i in range(du)]}

        eta = 1e-16
        dynamics.fit(vals, acts)

        prev_traj_dist = init_traj_dist(traj_states, traj_actions, dynamics, hyperparameters)
        traj_dist = prev_traj_dist
        prev_mu, prev_sigma = forward(prev_traj_dist, dynamics)
        prev_eta = -np.Inf
        min_eta = prev_eta

        _MAX_ITER = 5
        for iter in range(_MAX_ITER):

            # Collect samples in simulation
            for sample in range(10):
                s, a = get_sample(traj_dist,traj_no)
                push_sample(traj_no, s, a)
            vals, acts = getPreviousSA(traj_no, traj_states, traj_actions)

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
            # kl_div = calculate_KL_div(mu, prev_mu, traj_dist, prev_traj_dist)
            # print(kl_div)
            # if kl_div <= _THRESHOLD:
            #     break
            prev_traj_dist = traj_dist

        #Take initial sample
        samples = np.array([-np.random.multivariate_normal(mu[t], sigma[t], 1).flatten() for t in range(T)])
        commands = samples[:,28:]
        # raw_input()
        f = open('trajectories/target/Traj{}pred.txt'.format(traj_no+1), 'w')
        print('here')

        for act in commands:
            f.write("{}\n".format(" ".join(str(x) for x in act.flatten())))

        f.close()

    return samples

def get_sample(traj_dist,experiment_id):
    #Start a simulation and do the stuff
    functions = {}
    args = {}
    real_fun(functions)

    states = np.genfromtxt('trajectories/target/Traj'+str(experiment_id+1)+'state.txt', delimiter=',',dtype=np.float32)
    actions = np.genfromtxt('trajectories/target/Traj'+str(experiment_id+1)+'action.txt' ,dtype=np.float32)
    T,d = states.shape

    states = states[:,3:]
    states[np.isnan(states)] = 0
    actions[np.isnan(actions)] = 0

    #Create simulation
    # Sim = Simulation(function=functions["Traj{}".format(str(experiment_id+1))], args=[[0,0,3],0])
    # Sim.restart()

    f = open('trajectories/target/Traj{}pred.txt'.format(experiment_id+1),'w+')
    r = open('trajectories/target/Traj{}residuals.txt'.format(experiment_id+1),'w+')

    for timestep in range(T-1):
        # states[timestep,:] = normalize(state_norms[experiment_id],controller.getDif(Sim.cid,Sim.copter,Sim.target))
        old = actions[timestep,:].copy()
        # actions[timestep,:] = denormalize(action_norms[experiment_id], get_action(traj_dist,states[timestep,:],timestep)[0])
        # Sim.forward(actions[timestep,:].tolist())
        f.write(str(actions[timestep,:])+'\n')
        r.write(str(old-actions[timestep,:])+'\n')
        #Sim.forward()
        # Sim.sync()
    # print(vrep.simxStopSimulation(Sim.cid,vrep.simx_opmode_oneshot_wait))
    f.close()
    r.close()
    return states,actions


def normalize(norms,val):
    mean, dif = norms[0], norms[1]
    return (val - mean)/dif

def denormalize(norms,val):
    mean, dif = norms[0], norms[1]
    return (val * dif) + mean

def get_action(traj_dist, state, t):
    return -np.random.multivariate_normal(np.dot(traj_dist.K[t, :, :], state) + traj_dist.k[t, :], traj_dist.covar[t, :, :], 1)

def push_sample(traj_index,states,act):
    global iteration_state_repo
    global iteration_action_repo
    firstempty = getLast(iteration_state_repo[traj_index])
    iteration_state_repo[traj_index][firstempty] = states
    iteration_action_repo[traj_index][firstempty] = act

def calculate_KL_div(new_mu, prev_mu, cur_traj_dist, prev_traj_dist):
    """ Calculate KL divergence for two multivariate Gaussian distributions. """

    T, du, dx = cur_traj_dist.dimensions

    # (1 x T) matrix, div for each time step
    kl_div = np.zeros((1, T))

    for t in range(T):

        new_mu_t = new_mu[t,:]
        prev_mu_t = prev_mu[t,:]

        prev_cov = prev_traj_dist.covar[t,:,:]
        new_cov = cur_traj_dist.covar[t,:,:]
        new_inv_cov = cur_traj_dist.inv_cov[t,:,:]

        print(prev_cov.shape)
        print(new_cov.shape)
        print(new_inv_cov.shape)

        kl_div_t = 0.5 * (np.trace(new_inv_cov * prev_cov) +\
                   (new_mu_t - prev_mu_t).T.dot(new_inv_cov).dot(new_mu_t - prev_mu_t) - T + np.log(np.det(new_cov)) - np.log(np.det(prev_cov)))

        kl_div[t] = max(0, kl_div_t)

    # sum total kl_div over all time steps
    return np.sum(kl_div)

def compute_costs(traj_dist, eta, state, action, hyperparameters):
    """
    IN:
        traj_dist:      trajectory dist p(ut | xt)
        eta             (dual variable)
        state:          (T x dx)
        action:         (T x du)
        hyperparameters:    dict{'wu':  (1 x du) of weights for action
                                 'wx':  (1 x dx) of weights for state
    OUT:
        Hessian:        (T x (du + dx) x (du + dx)) matrix
        Jacobian:       (T x (du+dx)) matrix
        (both Hessian, Jacobian taken w.r.t. [xt ; ut]
    """
    T = traj_dist.dimensions[0]
    hessian, jacobian = get_jacobian_hessian(eta, state, action, hyperparameters)
    K = traj_dist.K
    k = traj_dist.k
    inv_cov = traj_dist.inv_covar

    for t in range(T-1, -1, -1):

        jacobian[t,:] += np.hstack([K[t, :, :].T.dot(inv_cov[t, :, :]).dot(k[t, :]),
                                      -inv_cov[t,:,:].dot(k[t,:])])

        hessian[t,:,:] += np.vstack([np.hstack([K[t,:,:].T.dot(inv_cov[t,:,:]).dot(K[t,:,:]),
                                                -K[t,:,:].T.dot(inv_cov[t,:,:])]),
                                    np.hstack([-inv_cov[t,:,:].dot(K[t,:,:]), inv_cov[t,:,:]])])
    return hessian, jacobian


def get_jacobian_hessian(eta, state, action, hyperparameters):
    """
    IN:
        eta
        state:          (T x dx)
        action:         (T x du)
        hyperparameters:    dict{'wu':  (1 x du) of weights for action
                                 'wx':  (1 x dx) of weights for state
    OUT:
        Hessian:        (T x (du + dx) x (du + dx)) matrix
        Jacobian:       (T x (du+dx)) matrix
        (both Hessian, Jacobian taken w.r.t. [xt ; ut]
    """

    wx = np.array(hyperparameters['wx'])
    wu = np.array(hyperparameters['wu'])

    T, du = action.shape
    dx = state.shape[1]
    jacobian = np.concatenate(( wx[:,np.newaxis].T * state, wu[:, np.newaxis].T * action), axis=1)

    lxx = np.diag(wx)
    luu = np.diag(wu)
    lux = np.zeros((dx, du))

    hessian = np.concatenate((np.concatenate((lxx, lux), axis=1), np.concatenate((lux.T, luu), axis=1)))
    hessian_final = np.tile(hessian, [T, 1, 1]) # For all trajectories
    return hessian_final / eta, jacobian / eta


def init_traj_dist(state, action, dynamics, hyperparameters):

    T, du = action.shape
    dx = state.shape[1]
    K = np.zeros((T, du, dx))
    k = np.zeros((T, du))
    inv_covar = np.zeros((T, du, du))
    covar = np.zeros((T, du, du))

    dx_slice = slice(dx)
    du_slice = slice(dx, dx+du) # slice out actions

    vt = np.zeros(dx)
    vtt = np.zeros((dx, dx))

    eta = 1e-20

    ctt, ct = get_jacobian_hessian(eta, state, action, hyperparameters)

    Fm = dynamics.Fm
    fv = dynamics.fv

    # backward pass
    for t in range(T-1, -1, -1):

        qtt = ctt[t, :, :] + Fm[t,:,:].T.dot(vtt).dot(Fm[t,:,:])
        qt = ct[t, :] + Fm[t,:, :].T.dot(vt + vtt.dot(fv[t,:]))

        # LU decomposition
        P, L, U = splinalg.lu(qtt[du_slice, du_slice])

        inv_covar[t, :, :] = qtt[du_slice, du_slice]
        covar[t, :, :] = sp.linalg.solve_triangular(
            U, splinalg.solve_triangular(L, np.eye(du), lower=True)
        )

        K[t, :, :] = -sp.linalg.solve_triangular(
            U, splinalg.solve_triangular(L, qtt[du_slice, dx_slice], lower=True)
        )
        k[t, :] = -sp.linalg.solve_triangular(
            U, splinalg.solve_triangular(L, qt[du_slice], lower=True)
        )

        vtt = qtt[dx_slice, dx_slice] + qtt[dx_slice, du_slice].dot(K[t, :, :])
        vt = qt[dx_slice] + qtt[dx_slice, du_slice].dot(k[t,:])
        vtt = 0.5* (vtt + vtt.T)

    return LinearGaussian(K, k, covar, inv_covar)


def forward(traj_dist, dynamics):

    # get dimensions of action, state matrices
    T, du, dx = traj_dist.dimensions

    dx_slice = slice(dx)
    # Initialize mu, sigma
    mu = np.zeros((T, dx + du))
    sigma = np.zeros((T, dx+du, dx + du))

    Fm = dynamics.Fm
    fv = dynamics.fv
    covar = dynamics.covar

    sigma[0, dx_slice, dx_slice] = dynamics.x0sigma
    mu[0, dx_slice] = dynamics.x0mu

    for t in range(T):

        sigma[t, :, :] = np.vstack([
            np.hstack([
                sigma[t, dx_slice, dx_slice],
                sigma[t, dx_slice, dx_slice].dot(traj_dist.K[t, :, :].T)
            ]),
            np.hstack([
                traj_dist.K[t, :, :].dot(sigma[t, dx_slice, dx_slice]),
                traj_dist.K[t, :, :].dot(sigma[t, dx_slice, dx_slice]).dot(
                    traj_dist.K[t, :, :].T) + traj_dist.covar[t, :, :]
                ])
            ])
        mu[t, :] = np.hstack([
            mu[t, dx_slice],
            traj_dist.K[t, :, :].dot(mu[t, dx_slice]) + traj_dist.k[t, :]
        ])

        if t < T - 1:
            sigma[t+1, dx_slice, dx_slice] = Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + covar[t, :, :] # Transition
            mu[t+1, dx_slice] = Fm[t, :, :].dot(mu[t, :]) + fv[t,:] #Transition

    return mu, sigma


def backward(state, action, dynamics, eta, hyperparameters):

    T, dx = state.shape
    du = action.shape[1]

    # Empty gaussian
    K = np.zeros((T, du, dx))
    k = np.zeros((T, du))
    inv_cov = np.zeros((T, du, du))
    cov = np.zeros((T, du, du))
    traj_dist = LinearGaussian(K, k, cov, inv_cov)

    dx_slice = slice(dx)
    du_slice = slice(dx, dx+du)
    eta0 = eta
    del_ = 1e-32
    Fm = dynamics.Fm
    fv = dynamics.fv

    linalgerr = True
    while linalgerr:

        linalgerr = False
        vxx = np.zeros((T, dx, dx))
        vx = np.zeros((T, dx))

        ctt, ct = compute_costs(traj_dist, eta, state, action, hyperparameters)
        for t in range(T-1, -1, -1):

            qtt = ctt[t, :, :]
            qt = ct[t, :]

            if t < T-1:
                qtt = qtt + Fm[t,:,:].T.dot(vxx[t+1,:,:]).dot(Fm[t, :, :])
                qt = qt + Fm[t,:,:].T.dot(vx[t+1, :] + vxx[t+1,:,:].dot(fv[t,:]))
            qtt = 0.5 * (qtt + qtt.T)

            try:
                # LU decomposition
                P, L, U = splinalg.lu(qtt[du_slice, du_slice])

                inv_cov[t, :, :] = qtt[du_slice, du_slice]
                cov[t, :, :] = sp.linalg.solve_triangular(
                    U, splinalg.solve_triangular(L, np.eye(du), lower=True)
                )

                K[t, :, :] = -sp.linalg.solve_triangular(
                    U, splinalg.solve_triangular(L, qtt[du_slice, dx_slice], lower=True)
                )
                k[t, :] = -sp.linalg.solve_triangular(
                    U, splinalg.solve_triangular(L, qt[du_slice], lower=True)
                )
                print('poop')

            except np.linalg.LinAlgError:
                linalgerr = True
                break

            traj_dist.inv_cov = inv_cov
            traj_dist.cov = cov
            traj_dist.K = K
            traj_dist.k = k

            vxx[t,:,:] = qtt[dx_slice, dx_slice] + qtt[dx_slice, du_slice].dot(K[t, :, :])
            vx[t,:] = qt[dx_slice] + qtt[dx_slice, du_slice].dot(k[t, :])
            vxx[t,:,:] = 0.5 * (vxx[t,:,:] + vxx[t,:,:].T)

        if linalgerr:
            old_eta = eta
            eta = eta0 + del_

            del_ *= 2  # Increase del_ exponentially on failure.
            if eta >= 1e16:
                if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                    raise ValueError('NaNs encountered in dynamics!')
                raise ValueError('Failed to find PD solution even for very \
                        large eta (check that dynamics and cost are \
                        reasonably well conditioned)!')

    return traj_dist, eta


def setup():
    global iteration_state_repo
    global iteration_action_repo
    iteration_state_repo = [[0]*100]*100
    iteration_action_repo = [[0]*100]*100

setup()

def getLast(list):
    i = 0
    while i < len(list)-1 and not type(list[i]) == int:
        i = i+1
    return i

def getPreviousSA(index,traj,act):
    global iteration_state_repo
    global iteration_action_repo

    if (type(iteration_state_repo[index][0]) == int):
        #Load in copy
        for x in range(3):
            iteration_state_repo[index][x] = traj
            iteration_action_repo[index][x] = act
    l = getLast(iteration_state_repo[index])
    out_states = np.asarray(iteration_state_repo[index][l-3:l])
    out_actions = np.asarray(iteration_action_repo[index][l - 3:l])
    return out_states, out_actions
