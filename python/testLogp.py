from scipy.stats import multivariate_normal
import numpy as np

def neglogp(action,mean,std):
    return 0.5*np.sum(np.power(action-mean,2)/np.power(std,2)) \
            + np.log(np.sqrt(np.prod(np.power(2*np.pi*std,2))))

def neglogp2(action,mean,std):
    return -np.log(gaussian(action,mean,std))

def logp2(action,mean,std):
    return np.log(gaussian(action,mean,std))

def gaussian(action,mean,std):
    return np.exp(-0.5*np.sum(np.power(action-mean,2)/np.power(std,2),axis=-1))\
           /np.sqrt(np.prod(np.power(2*np.pi*std,2),axis=-1))

def gaussian2(action,mean,std):
    return np.exp(logp2(action,mean,std))

def gaussian3(action,mean,std):
    cov = np.diag(std**2)	
    k = action.shape[0]
    p1 = np.exp(-0.5*k*np.log(2*np.pi))
    p2 = np.power(np.linalg.det(cov),-0.5)
    dev = action - mean
    p3 = np.exp(-0.5*np.dot(np.dot(dev.transpose(),np.linalg.inv(cov)),dev))
    return p1*p2*p3



'''
action = np.array(
              [1,2,2,2,3,6,32,12,44,1,2,3,3,54,6,4],
              dtype=np.float64).reshape((4,4))
mean = np.array(
            [1.1,2.1,3,1.8,2.5,5,30,15,40,3,2.2,2.7,3.1,52,5.8,4.1],
            dtype=np.float64).reshape((4,4))
std = np.array(
           [0.1,0.1,0.2,0.1,0.2,1,3,4,8,3,2,1,0.2,1,0.3,0.4],
           dtype=np.float64).reshape((4,4))
'''
action = np.array(
              [0.0771532, -0.0921249, 0.155535, 0.110629, -0.0750895, 0.0490313, 0.0416572, -0.0489283, 0.117888, 0.0104795, 0.0261741, -0.193333, -0.199992, -0.119793, -0.00147295, 0.0347881, -0.0862036, -0.0163956, -0.011512, -0.0051254, 0.0744867, -0.0815146, 0.00417948, 0.0575905, 0.0776339, -0.174205, -0.00825691, 0.0330421, -0.163067, 0.0111632, 0.104023, -0.0618963, -0.171926, -0.0930436, 0.0505307, -0.0134338, -0.100431, -0.133168, 0.146748, 0.120166],
              dtype=np.float64).reshape((10,4))
mean = np.array(
            [-0.0125206, -0.019085, 0.0512585, -0.0105176,
             -0.0125206, -0.019085, 0.0512585, -0.0105176,
             -0.0125206, -0.019085, 0.0512585, -0.0105176,
             -0.0125206, -0.019085, 0.0512585, -0.0105176,
             -0.0125207, -0.019085, 0.0512585, -0.0105176,
             -0.0125207, -0.0190849, 0.0512585, -0.0105176,
             -0.0125207, -0.0190849, 0.0512585, -0.0105176,
             -0.0125207, -0.0190849, 0.0512585, -0.0105176,
             -0.0125207, -0.0190849, 0.0512585, -0.0105176,
             -0.0125207, -0.0190849, 0.0512585, -0.0105177,
             ],
            dtype=np.float64).reshape((10,4))
std = np.array(
           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
           dtype=np.float64).reshape((10,4))


#res = np.exp(logp2(action,mean,std))
#res = logp2(action,mean,std)
#res = np.exp(-neglogp(action,mean,std))
#res = gaussian3(action[0,:],mean[0,:],std[0,:])
res1 = gaussian(action[0,:],mean[0,:],std[0,:])
res2 = gaussian2(action[0,:],mean[0,:],std[0,:])
#res3 = gaussian3(action,mean,std)
res4 = multivariate_normal.pdf(action,mean,std**2)

print("gaussian1 pdf:\n",res1)
print("gaussian2 pdf:\n",res2)
print("gaussian3 pdf:\n",res3)
print("scipy normal pdf:\n",res4)
