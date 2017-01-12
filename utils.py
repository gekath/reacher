import vrep
import numpy as np
from sklearn import preprocessing
SYNC = True

def connect():

    cid=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
    if cid != -1:
        print ('Connected to V-REP remote API serv'
               '\er, client id: %s' % cid)
        vrep.simxStartSimulation( cid, vrep.simx_opmode_oneshot )
        if SYNC:
            vrep.simxSynchronous( cid, True )
    else:
        print ('Failed connecting to V-REP remote API server')
        exit()
    return cid

def target_move(cid,obj, firstPass, function, *args):
    args = cleanse(args[0])
    pos = function(args)
    vrep.simxSetObjectPosition(cid,obj,-1,pos,mode(firstPass))

def mode(firstPass):
    # global firstPass
    return vrep.simx_opmode_buffer if firstPass else vrep.simx_opmode_streaming

def cleanse(X):
    pos, EitherFunOrVal = X[0],X[1]
    if (type(EitherFunOrVal) == type(cleanse)):
        EitherFunOrVal = EitherFunOrVal()
    return [pos,EitherFunOrVal]

def controller_motor(clientID, target_handle, self_handle,joint_handles, firstPass):
    # global firstPass
    global joint_target_velocities
    global motor_mask
    global u

    if (firstPass):
        joint_target_velocities = np.ones(len(joint_handles)) * 10000.0
        u = [1]*len(joint_handles)
        motor_mask = [0]*len(joint_handles)
        print("OMG FIRSTPASS")
    #-- Decide of the motor velocities:
    #Grab target
    #error, target_pos = vrep.simxGetObjectPosition(cid,target_handle,self_handle,mode())

    #Grab joint angles
    q = np.zeros(len(joint_handles))
    dq = np.zeros(len(joint_handles))
    for ii,joint_handle in enumerate(joint_handles):
        if (ii < 2):
            continue
        # get the joint angles
        _, q[ii] = vrep.simxGetJointPosition(clientID,
                joint_handle,
                vrep.simx_opmode_oneshot_wait)
        if _ !=0 : raise Exception()
        # get the joint velocity
        _, dq[ii] = vrep.simxGetObjectFloatParameter(clientID,
                joint_handle,
                2012, # ID for angular velocity of the joint
                vrep.simx_opmode_oneshot_wait)
        if _ !=0 : raise Exception()

        # get the current joint torque
        _, torque = vrep.simxGetJointForce(clientID, joint_handle, vrep.simx_opmode_oneshot_wait)
        if _ !=0 : raise Exception()

        joint_target_velocities[ii] = 5
        u[ii] = np.random.uniform(-500,500)
        motor_mask[2] = 1
        # if force has changed signs,
        # we need to change the target velocity sign

    vrep.simxPauseCommunication(clientID,1);
    for ii,joint_handle in enumerate(joint_handles):
        if np.sign(torque) * np.sign(u[ii]) < 0:
            joint_target_velocities[ii] = joint_target_velocities[ii] * -1
        if (motor_mask[ii]):
            vrep.simxSetJointTargetVelocity(clientID, joint_handle, joint_target_velocities[ii], vrep.simx_opmode_oneshot)
            vrep.simxSetJointForce(clientID,
                joint_handle,
                abs(u[ii]), # force to apply
                vrep.simx_opmode_oneshot)
        if _ !=0 : raise Exception()
    vrep.simxPauseCommunication(clientID,0);
    firstPass = False

    return firstPass

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def get_flat_files():
    """
    This is not the same as the version seen in traj_opt, this returns a [N*T,9] and [N*T,4] (N is num trajectories)
    :return:
    """
    T = 39
    dx = 31
    du = 7
    N = 83

    states = np.zeros((T*N, dx))
    actions = np.zeros((T*N, du))
    # state_normalizers = np.zeros((1, 2))
    # action_normalizers = np.zeros((1, 2))
    directory = 'trajectories/target/'
    names = ['Traj' + str(x) for x in range(1,N)]
    vars = range(N-1)
    for traj_no in vars:
        var = names[traj_no]
        file = var + 'state.txt'
        state = np.array(
            np.genfromtxt(directory + file, delimiter=','))
        # state = state[1:]
        state_mean = np.mean(state, axis=0)
        state_max_diff = np.max(state, axis=0) - np.min(state, axis=0)
        # state_normalizers = np.append(state_normalizers, np.array([[np.mean(state),(np.max(state)-np.min(state))]]),axis =0)
        # state = (state - state_normalizers[-1][0]) / state_normalizers[-1][1]
        # state = (state - state_mean) / state_max_diff
        # print(np.array([state])[0,:-1,:].shape)
        # print(traj_no)
        states[traj_no*(T):(traj_no+1)*(T),:] = np.array([state])[0,:-1,:]

        file = var + 'action.txt'
        action = np.array(np.genfromtxt(directory + file, delimiter=' '))
        # action = action[1:] #Keep only second and after
        # action_normalizers = np.append(action_normalizers, np.array([[np.mean(action),(np.max(action)-np.min(action))]]),axis =0)
        # action = (action- action_normalizers[-1][0]) / action_normalizers[-1][1]
        action_mean = np.mean(action, axis=0)
        action_max_diff = np.max(action, axis=0) - np.min(action, axis=0)
        # action = (action - action_mean) / action_max_diff

        actions[traj_no*T:(traj_no+1)*T,:] = np.array([action])[0,:,:]

    # States is 27xTx9
    # Actions is 27xTx4
    # actions = actions[1:]
    # state_normalizers = state_normalizers[1:]
    # action_normalizers = action_normalizers[1:]
    return states[:,3:], actions

    # total = np.zeros((T*N, dx + du))
    # total[:,:dx] = states
    # total[:,dx:] = actions
    # return total

def get_all_files():
    """
    This is not the same as the version seen in traj_opt, this returns a [N*T,9] and [N*T,4] (N is num trajectories)
    :return:
    """
    T = 39
    dx = 31
    du = 7
    N = 20

    states = np.zeros((N, T, dx))
    actions = np.zeros((N,T, du))
    # state_normalizers = np.zeros((1, 2))
    # action_normalizers = np.zeros((1, 2))
    directory = 'trajectories/target/'
    names = ['Traj' + str(x) for x in range(1, N)]
    for traj_no in range(N-1):
        var = names[traj_no]
        file = var + 'state.txt'
        state = np.array(
            np.genfromtxt(directory + file, delimiter=','))
        # state = state[1:]
        state_mean = np.mean(state, axis=0)
        state_max_diff = np.max(state, axis=0) - np.min(state, axis=0)
        # state_normalizers = np.append(state_normalizers, np.array([[np.mean(state),(np.max(state)-np.min(state))]]),axis =0)
        # state = (state - state_normalizers[-1][0]) / state_normalizers[-1][1]
        # state = (state - state_mean) / state_max_diff
        states[traj_no,:,:] = np.array([state])[0,:-1,:]

        file = var + 'action.txt'
        action = np.array(np.genfromtxt(directory + file, delimiter=' '))
        # action = action[1:] #Keep only second and after
        # action_normalizers = np.append(action_normalizers, np.array([[np.mean(action),(np.max(action)-np.min(action))]]),axis =0)
        # action = (action- action_normalizers[-1][0]) / action_normalizers[-1][1]
        action_mean = np.mean(action, axis=0)
        action_max_diff = np.max(action, axis=0) - np.min(action, axis=0)
        # action = (action - action_mean) / action_max_diff

        actions[traj_no,:,:] = np.array([action])[0,:,:]

    # States is 27xTx9
    # Actions is 27xTx4
    # actions = actions[1:]
    # state_normalizers = state_normalizers[1:]
    # action_normalizers = action_normalizers[1:]
    return states[:,:,3:], actions

    # total = np.zeros((T*N, dx + du))
    # total[:,:dx] = states
    # total[:,dx:] = actions
    # return total

def compute_jacobian(state, weights):
    '''
    IN:
        state:      (T x dn)
        weights:    (1 x dn)
    OUT:
        jacobian: (T x dn)
    '''

    pass

def compute_hessian(state1, state2, weight1, weight2):
    '''
    IN:
        state:  (T x dx)
        action: (T x du)
        hyperparameters:
            w_u:    (1 x dx)
            w_x:    (1 x du)
    OUT:
        hessian: (T x (dx + du) x (dx + du))
    '''

    pass
