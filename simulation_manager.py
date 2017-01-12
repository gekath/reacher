import sys
import vrep
import time
from functools import reduce
from function_names import *
from control.ilqr import *
from utils import *
from nn import *
from lqr import *
firstPass = True
runtime = 0.0

ONE_SHOT = vrep.simx_opmode_oneshot_wait
vrep_mode = vrep.simx_opmode_oneshot

def getTrajectories():
    print ('Program started')

    #Start Grabbing Handles
    functions = {}
    args = {}
    real_fun(functions)
    #1-15 are examples along a sphere, one at top of sphere, one in middle, one in bottom,
    #and then three rings at centred .25d, .5d, .75d with radius = 2

    k = functions.keys()
    k.sort()

    iter = 1
    for key in k:
        firstPass = True
        state_file = open('trajectories/target/{}.txt'.format(key+"state"),'w+')
        action_file = open('trajectories/target/{}.txt'.format(key+"action"),'w+')

        vrep.simxFinish(-1)  # just in case, close all opened connections
        time.sleep(1)
        cid = connect()
        time.sleep(1)

        vrep.simxLoadScene(cid,'scene/arm_2.ttt',0,mode())
        dt = .01
        vrep.simxSetFloatingParameter(cid,
                vrep.sim_floatparam_simulation_time_step,
                dt, # specify a simulation time step
                vrep.simx_opmode_oneshot)
        vrep.simxStartSimulation(cid,vrep.simx_opmode_streaming)
        runtime = 0

        joint_names = ['redundantRob_joint'+str(x) for x in range(1,8)]
        # print(joint_names)
        # # joint target velocities discussed below
        # joint_target_velocities = np.ones(len(joint_names)) * 10000.0
        #
        # # get the handles for each joint and set up streaming
        joint_handles = [vrep.simxGetObjectHandle(cid,
            name, vrep.simx_opmode_oneshot_wait)[1] for name in joint_names]
        # print(joint_handles)
        # # get handle for target and set up streaming
        _, target_handle = vrep.simxGetObjectHandle(cid,
                        'redundantRob_manipSphere', vrep.simx_opmode_oneshot_wait)
        #
        _, self_handle = vrep.simxGetObjectHandle(cid,'Rectangle',vrep.simx_opmode_oneshot_wait)
        #
        _, target_pos = vrep.simxGetObjectPosition(cid,
                                                   target_handle, -1, vrep.simx_opmode_oneshot_wait)

        #This prepared the data
        real_args(args, target_pos)
        print(key)

        while(runtime < 0.4):
            target_move(cid,target_handle, firstPass, functions[key], args[key])
            #vrep.simxSynchronous(cid, False)
            vrep.simxGetPingTime(cid)
            firstPass = controller_motor(cid,target_handle,self_handle, joint_handles, firstPass)

            commands = [vrep.simxGetJointForce(cid, joint, vrep.simx_opmode_oneshot_wait)[1] for joint in
                        joint_handles]

            joint_pos = [vrep.simxGetObjectPosition(cid, joint, vrep.sim_handle_parent, vrep.simx_opmode_oneshot_wait)[1] for joint
                         in joint_handles]

            joint_vel = [vrep.simxGetObjectFloatParameter(cid, joint, 2012, vrep.simx_opmode_blocking)[1]
                         for joint in joint_handles]

            target_rel = vrep.simxGetObjectPosition(cid, target_handle, joint_handles[-1],
                                                    vrep.simx_opmode_oneshot_wait)[1]

            joint_state = np.append(np.asarray(joint_pos).flatten(),target_rel + joint_vel)
            # joint_state = np.append(np.asarray(joint_pos).flatten(),joint_vel)
            # print(np.asarray(joint_state).shape)
            # print(joint_state)

            state_file.write("{}\n".format(",".join(
                [str(x) for x in joint_state])))
            action_file.write(
                "{}\n".format(" ".join([str(x)[1:-2] for x in commands])))
            runtime += dt

            vrep.simxSynchronousTrigger(cid)
            firstPass = False

        vrep.simxStopSimulation(cid, vrep.simx_opmode_streaming)
        vrep.simxSynchronousTrigger(cid)
        vrep.simxFinish(cid)
        firstPass = True
        print("meow")

def sendToVrep(traj_no):

    functions = {}
    args = {}
    real_fun(functions)
    # 1-15 are examples along a sphere, one at top of sphere, one in middle, one in bottom,
    # and then three rings at centred .25d, .5d, .75d with radius = 2
    key = 'Traj{}'.format(traj_no)

    state_normalizers = np.zeros((1, 2))
    action_normalizers = np.zeros((1, 2))

    states = np.genfromtxt('trajectories/target/Traj{}state.txt'.format(traj_no), delimiter=',', dtype=np.float32)[1:]

    actions = np.genfromtxt('trajectories/target/Traj{}pred.txt'.format(traj_no),delimiter=' ', dtype=np.float32)[1:]

    # states, actions = get_flat_files()

    target_vel = states[:,24:]
    # states = states[:,:7]


    T, du = np.asarray(actions).shape  # Close all threads, in case
    # _, dx = np.asarray(states).shape

    # ilqr = Control(traj_no)
    # X, U, cost = ilqr.ilqr(states[0], actions)

    vrep.simxFinish(-1)
    cid = connect()
    # vrep.simxLoadScene(cid, 'scene/arm_no_control.ttt', 0, mode())
    dt = .01
    vrep.simxSetFloatingParameter(cid,
                                  vrep.sim_floatparam_simulation_time_step,
                                  dt,  # specify a simulation time step
                                  vrep.simx_opmode_oneshot)
    vrep.simxStartSimulation(cid, vrep.simx_opmode_streaming)


    joint_names = ['redundantRob_joint' + str(x) for x in range(1, 8)]
    joint_handles = [vrep.simxGetObjectHandle(cid,
                                              name,
                                              vrep.simx_opmode_oneshot_wait)[1]
                     for name in joint_names]


    # # get handle for target and set up streaming
    _, target_handle = vrep.simxGetObjectHandle(cid,
                                                'redundantRob_manipSphere',
                                                vrep.simx_opmode_oneshot_wait)
    #
    _, self_handle = vrep.simxGetObjectHandle(cid, 'Rectangle',
                                              vrep.simx_opmode_oneshot_wait)
    #
    _, target_pos = vrep.simxGetObjectPosition(cid,
                                               target_handle, -1,
                                               vrep.simx_opmode_oneshot_wait)

    real_args(args, target_pos)

    target_move(cid, target_handle, False, functions[key],
                           args[key])

    for t in range(T):
        #
        # for j in range(du):
        #     if np.sign(actions[t,j]) * target_vel[t,j] < 0:
        #         target_vel[j] = target_vel[j] * -1

        [vrep.simxSetJointTargetVelocity(cid, joint_handles[j], target_vel[t,j],
                                         vrep.simx_opmode_streaming) for j in range(du)]
        [vrep.simxSetJointForce(cid, joint_handles[j], actions[t,j],
                                vrep.simx_opmode_streaming) for j in range(du)]
        vrep.simxSynchronousTrigger(cid)
        # raw_input()
        print('here')

    vrep.simxSynchronousTrigger(cid)
    vrep.simxStopSimulation(cid, vrep.simx_opmode_streaming)
    vrep.simxFinish(cid)



def learn_policy(traj_no):
    # functions = {}
    # args = {}
    # real_fun(functions)
    #
    # states = np.genfromtxt('Trajectories/Traj1state.txt', delimiter=',',dtype=np.float32)
    # actions = np.genfromtxt('Trajectories/Traj1action.txt' ,dtype=np.float32)
    states, actions = get_flat_files()

    dx = 28

    # total[total < 1e-3] = 0
    states[np.isnan(states)] = 0
    actions[np.isnan(actions)] = 0

    # scaler = preprocessing.MinMaxScaler()
    # scaled_total = scaler.fit_transform(total)
    # scaled_total = total

    # states = scaled_total[:, :dx]
    # actions = scaled_total[:, dx:]

    #Create simulation
    # Sim = Simulation(function=functions["Traj1"], args=[[0,0,3],0])
    nn = Net()
    states, actions = shuffle_in_unison(states,actions)

    T,d = states.shape
    du = 7
    num_train = int(np.floor(T * .70))
    train_states = states[:num_train,:]
    train_actions = actions[:num_train,:]
    val_states, val_actions =- states[num_train:,:], actions[num_train:,:]
    sess, out, feed_me, keep_prob, momentum = nn.train(train_states,train_actions,40,val_states,val_actions)
    # Sim.restart()


    key = 'Traj{}'.format(traj_no)
    action_file = open('trajectories/target/{}.txt'.format(key + "actionpred"),
                       'w+')
    state1 = np.genfromtxt('trajectories/target/Traj{}state.txt'.format(traj_no), delimiter=',',dtype=np.float32)
    T, dx = state1.shape

    # noise = np.random.normal(0, 1e-5, (T, dx))
    # print(state1)
    # print(noise)
    # raw_input()
    # state1 = state1 + noise

    for t in range(T):
        val = np.zeros((1,dx-3))
        val[0,:] = state1[t,3:]
        pred_action = sess.run([out], feed_dict={feed_me:val, keep_prob:1})
        pred_action = np.asarray(pred_action,dtype=np.float32)[0,:,:]
        action_file.write(
            "{}\n".format(" ".join([str(x)[1:-2] for x in pred_action.flatten()])))

# def learn_policy(traj_no):
#     functions = {}
#     args = {}
#     real_fun(functions)
#     key = 'Traj{}'.format(traj_no)
#     #
#     # states = np.genfromtxt('Trajectories/Traj1state.txt', delimiter=',',dtype=np.float32)
#     # actions = np.genfromtxt('Trajectories/Traj1action.txt' ,dtype=np.float32)
#     states, actions = get_flat_files()
#
#     dx = 28
#
#     # total[total < 1e-3] = 0
#     states[np.isnan(states)] = 0
#     actions[np.isnan(actions)] = 0
#
#     # scaler = preprocessing.MinMaxScaler()
#     # scaled_total = scaler.fit_transform(total)
#     # scaled_total = total
#
#     # states = scaled_total[:, :dx]
#     # actions = scaled_total[:, dx:]
#
#     #Create simulation
#     # Sim = Simulation(function=functions["Traj1"], args=[[0,0,3],0])
#     nn = Net()
#     states, actions = shuffle_in_unison(states,actions)
#
#     T,d = states.shape
#     du = 7
#     num_train = int(np.floor(T * .70))
#     train_states = states[:num_train,:]
#     train_actions = actions[:num_train,:]
#     val_states, val_actions =- states[num_train:,:], actions[num_train:,:]
#     sess, out, feed_me, keep_prob, momentum = nn.train_2_hidden(train_states,train_actions,40,val_states,val_actions)
#     # Sim.restart()
#
#
#     vrep.simxFinish(-1)
#     cid = connect()
#     vrep.simxLoadScene(cid, 'scene/arm_no_control.ttt', 0, mode())
#     dt = .01
#     vrep.simxSetFloatingParameter(cid,
#                                   vrep.sim_floatparam_simulation_time_step,
#                                   dt,  # specify a simulation time step
#                                   vrep.simx_opmode_oneshot)
#     vrep.simxStartSimulation(cid, vrep.simx_opmode_streaming)
#
#     joint_names = ['redundantRob_joint' + str(x) for x in range(1, 8)]
#     joint_handles = [vrep.simxGetObjectHandle(cid,
#                                               name,
#                                               vrep.simx_opmode_oneshot_wait)[1]
#                      for name in joint_names]
#
#     # # get handle for target and set up streaming
#     _, target_handle = vrep.simxGetObjectHandle(cid,
#                                                 'redundantRob_manipSphere',
#                                                 vrep.simx_opmode_oneshot_wait)
#     #
#     _, self_handle = vrep.simxGetObjectHandle(cid, 'Rectangle',
#                                               vrep.simx_opmode_oneshot_wait)
#     #
#     _, target_pos = vrep.simxGetObjectPosition(cid,
#                                                target_handle, -1,
#                                                vrep.simx_opmode_oneshot_wait)
#
#     real_args(args, target_pos)
#
#     controller.target_move(cid, target_handle, False, functions[key],
#                            args[key])
#
#     key = 'Traj{}'.format(traj_no)
#     action_file = open('trajectories/target/{}.txt'.format(key + "actionpred"),
#                        'w+')
#     state_file = open('trajectories/target/{}.txt'.format(key + "statepred"), 'w+')
#     state1 = np.genfromtxt('trajectories/target/Traj{}state.txt'.format(traj_no), delimiter=',',dtype=np.float32)
#     T, dx = state1.shape
#
#     val = np.zeros((1, dx-3))
#     val[0,:] = state1[2,3:]
#     [vrep.simxSetObjectPosition(cid, joint_handles[i], vrep.sim_handle_parent, val[0,(i-1)*3:i*3],vrep.simx_opmode_oneshot_wait) for i in range(1,len(joint_handles))]
#     [vrep.simxSetJointTargetVelocity(cid, joint_handles[i], val[0,21+i], vrep.simx_opmode_oneshot_wait) for i in range(len(joint_handles))]
#
#     for t in range(T):
#         pred_action = sess.run([out], feed_dict={feed_me:val, keep_prob:1})
#         pred_action = np.asarray(pred_action,dtype=np.float32)[0,:,:]
#         print(pred_action)
#
#         state_file.write('{}\n'.format(",".join([str(x) for x in val.flatten()])))
#         action_file.write(
#             "{}\n".format(" ".join([str(x)[1:-2] for x in pred_action.flatten()])))
#
#
#         [vrep.simxSetJointForce(cid, joint_handles[j], pred_action.flatten()[j],
#                             vrep.simx_opmode_streaming) for j in range(du)]
#
#         joint_pos = [
#             vrep.simxGetObjectPosition(cid, joint, vrep.sim_handle_parent,
#                                        vrep.simx_opmode_oneshot_wait)[1] for
#             joint
#             in joint_handles]
#
#         joint_vel = [vrep.simxGetObjectFloatParameter(cid, joint, 2012,
#                                                       vrep.simx_opmode_blocking)[
#                          1]
#                      for joint in joint_handles]
#
#         target_rel = \
#         vrep.simxGetObjectPosition(cid, target_handle, joint_handles[-1],
#                                    vrep.simx_opmode_oneshot_wait)[1]
#
#         joint_state = np.append(np.asarray(joint_pos).flatten(),
#                                 target_rel + joint_vel)
#
#         val[0,:] = joint_state[3:]
#         vrep.simxSynchronousTrigger(cid)
#         raw_input()



def get_sample_distribution():

    # functions = {}
    # args = {}
    # real_fun(functions)
    # 1-15 are examples along a sphere, one at top of sphere, one in middle, one in bottom,
    # and then three rings at centred .25d, .5d, .75d with radius = 2
    key = 'Traj{}'.format(traj_no)

    # state_normalizers = np.zeros((1, 2))
    # action_normalizers = np.zeros((1, 2))
    # states = np.genfromtxt('trajectories/target/Traj{}state.txt'.format(traj_no), delimiter=',', dtype=np.float32)[1:]
    # actions = np.genfromtxt('trajectories/target/Traj{}action.txt'.format(traj_no),delimiter=' ', dtype=np.float32)[1:]

    states, actions = get_all_files()
    states[isnan(states)] = 0
    actions[isnan(actions)] = 0
    # print(states.shape)
    # print(actions.shape)

    N, T, du = np.asarray(actions).shape  # Close all threads, in case
    _, _, dx = np.asarray(states).shape

    #
    # ilqr = Control(traj_no, max_iter=3)
    # X, U, cost = ilqr.ilqr(states[0,:], actions)

    samples = sample_dist(states, actions)
    raw_input()
    target_vel = X[:,24:]

    vrep.simxFinish(-1)
    cid = connect()
    vrep.simxLoadScene(cid, 'scene/arm_no_control.ttt', 0, mode())
    dt = .01
    vrep.simxSetFloatingParameter(cid,
                                  vrep.sim_floatparam_simulation_time_step,
                                  dt,  # specify a simulation time step
                                  vrep.simx_opmode_oneshot)
    vrep.simxStartSimulation(cid, vrep.simx_opmode_streaming)


    joint_names = ['redundantRob_joint' + str(x) for x in range(1, 8)]
    joint_handles = [vrep.simxGetObjectHandle(cid,
                                              name,
                                              vrep.simx_opmode_oneshot_wait)[1]
                     for name in joint_names]


    # # get handle for target and set up streaming
    _, target_handle = vrep.simxGetObjectHandle(cid,
                                                'redundantRob_manipSphere',
                                                vrep.simx_opmode_oneshot_wait)
    #
    _, self_handle = vrep.simxGetObjectHandle(cid, 'Rectangle',
                                              vrep.simx_opmode_oneshot_wait)
    #
    _, target_pos = vrep.simxGetObjectPosition(cid,
                                               target_handle, -1,
                                               vrep.simx_opmode_oneshot_wait)

    real_args(args, target_pos)

    target_move(cid, target_handle, False, functions[key],
                           args[key])

    for t in range(T):
        #
        # for j in range(du):
        #     if np.sign(actions[t,j]) * target_vel[t,j] < 0:
        #         target_vel[j] = target_vel[j] * -1

        [vrep.simxSetJointTargetVelocity(cid, joint_handles[j], target_vel[t,j],
                                         vrep.simx_opmode_streaming) for j in range(du)]
        [vrep.simxSetJointForce(cid, joint_handles[j], actions[t,j],
                                vrep.simx_opmode_streaming) for j in range(du)]
        vrep.simxSynchronousTrigger(cid)
        raw_input()
        print('here')

    vrep.simxSynchronousTrigger(cid)
    vrep.simxStopSimulation(cid, vrep.simx_opmode_streaming)
    vrep.simxFinish(cid)

if __name__ == "__main__":

    traj_no = 1
    if len(sys.argv) > 1:
        traj_no  = sys.argv[1]

    print("kumomon")
    print("kumamon wishes you good luck on stuff")
    # learn_policy(traj_no)
    sendToVrep(traj_no)
    # getTrajectories()
    # sampleDist(traj_no)

    # get_sample_distribution()