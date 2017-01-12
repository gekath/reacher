import vrep
import numpy as np
import numpy as np
from math import *
import ctypes

# firstPass = True
propellerScripts = [0]*4
def cleanse(X):
    pos, EitherFunOrVal = X[0],X[1]
    if (type(EitherFunOrVal) == type(cleanse)):
        EitherFunOrVal = EitherFunOrVal()
    return [pos,EitherFunOrVal]

def mode(firstPass):
    # global firstPass
    return vrep.simx_opmode_buffer if firstPass else vrep.simx_opmode_streaming

def simGetObjectMatrix(cid,obj,relative, firstPass):
    err, pos = vrep.simxGetObjectPosition(cid,obj,-1,mode(firstPass))
    x,y,z = pos
    print(err)
    print("Values are {} {} {} {}".format(x,y,z,pos))
    err, angles = vrep.simxGetObjectOrientation(cid,obj,-1,mode(firstPass))
    a,b,g = angles
    print(err)
    print("Angles are {} {} {} {}".format(a,b,g,angles))

    op = np.array([[0]*4]*4, dtype =np.float64)
    A = float(cos(a))
    B = float(sin(a))
    C = float(cos(b))
    D = float(sin(b))
    E = float(cos(g))
    F = float(sin(g))
    AD = float(A*D)
    BD = float(B*D)
    op[0][0]=float(C)*E
    op[0][1]=-float(C)*F
    op[0][2]=float(D)
    op[1][0]=float(BD)*E+A*F
    op[1][1]=float(-BD)*F+A*E
    op[1][2]=float(-B)*C
    op[2][0]=float(-AD)*E+B*F
    op[2][1]=float(AD)*F+B*E
    op[2][2]=float(A)*C
    op[0][3]=float(x)
    op[1][3]=float(y)
    op[2][3]=float(z)
    return op[0:3,:]

def getDif(cid,copter,target):
    copter_pos = vrep.simxGetObjectPosition(cid, copter, -1, mode(firstPass))[1]
    copter_vel = vrep.simxGetObjectVelocity(cid, copter, mode(firstPass))[1]
    copter_orientation = vrep.simxGetObjectOrientation(cid,copter,-1,mode(firstPass))[1]
    target_pos = vrep.simxGetObjectPosition(cid, target, -1, mode(firstPass))[1]
    target_vel = vrep.simxGetObjectVelocity(cid, target, mode(firstPass))[1]

    return np.asarray([(-np.asarray(copter_pos) + np.asarray(target_pos)),(-np.asarray(copter_vel) + np.asarray(target_vel)),np.asarray(copter_orientation)]).flatten()


