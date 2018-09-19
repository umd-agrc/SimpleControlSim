import re
import os.path
import numpy as np
import numpy.linalg as nplin

NUM_INPUTS = 4
NUM_STATES = 12

def parseNDArrayFile(filepath):
    d=None
    with open(filepath) as f:
        i=0
        for line in map(str.strip,f):
            if i==0:
                i = i + 1
                continue
            l = re.split(',\s*(?![^()[]]*[\)\]])',re.sub('[\[\]\s+]','',line).strip(','))
            d = np.array([float(x) for x in l])
            i = i + 1
    return d

def parseNDArrayVectorFile(filepath):
    d={}
    with open(filepath) as f:
        i=0
        for line in map(str.strip,f):
            if i==0:
                i = i + 1
                continue
            l = re.split(',\s*(?![^()[]]*[\)\]])',re.sub('[\[\]\s+]','',line).strip(','))
            d[i] = np.array([float(x) for x in l])
            i = i + 1
    return d

X_u   =  -0.27996
Y_v   =  -0.22566
Z_w   =  -1.2991
L_p   =  -2.5110
M_q   =  -2.4467
N_r   =  -0.4948
X_th  =  -10.067
Y_ph  =   9.8648
L_ph  =  -21.358
M_th  =  -18.664
ph_p  =   0.9655
th_q  =   0.9634
ps_r  =   0.6748
Z_thr =  -39.282
L_la  =   11.468
M_lo  =   9.5711
N_ya  =   3.5647
ph_la =   0.0744
th_lo =   0.0594
ps_ya =   0.0397

dynamics = np.array([
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, X_u, 0, 0, 0, 0, 0, 0, X_th, 0, 0, 0, 0, 0,
    0, 0, 0, 0, Y_v, 0, 0, 0, 0, Y_ph, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, Z_w, 0, 0, 0, 0, 0, 0, 0, 0, 0, Z_thr,
    0, 0, 0, 0, 0, 0, L_p, 0, 0, L_ph, 0, 0, L_la, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, M_q, 0, 0, M_th, 0, 0, M_lo, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, N_r, 0, 0, 0, 0, 0, N_ya, 0,
    0, 0, 0, 0, 0, 0, ph_p, 0, 0, 0, 0, 0, ph_la, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, th_q, 0, 0, 0, 0, 0, th_lo, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, ps_r, 0, 0, 0, 0, 0, ps_ya,0]).reshape(
            (NUM_STATES,NUM_STATES+NUM_INPUTS))

gainspath = os.path.join('..','data','lqr','arr','gains.ndarray')
statepath = os.path.join('..','data','lqr','arr','state.ndarray')

gains = np.reshape(parseNDArrayFile(gainspath),(NUM_INPUTS,NUM_STATES))
statevec = parseNDArrayVectorFile(statepath)
state = np.reshape(
            np.concatenate([statevec[k] for k in statevec]),
            (len(statevec),2*NUM_STATES))
stateError = state[:,0:NUM_STATES] - state[:,NUM_STATES:]
action = np.dot(gains,stateError.transpose()).transpose()
dState = np.dot(dynamics,np.concatenate([state[2,NUM_STATES:],action[2,:]]))
print(dState)
#nextstate = state[0,NUM_STATES:] + 0.01*np.dot(dynamics,np.concatenate([stateError[0,:],action[0,:]]))
#print(nextstate)
