# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:29:55 2018

@author: Matjaz
"""
from numpy import array as na
from numpy import eye, zeros
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt


def nvect(*argv):
    return np.array([argv]).T


cmap = plt.get_cmap("tab20c")

""" process model """
u=10.#m/s^2
deltaT=1e-2

# system state
F=na([[1, deltaT], [0, 1]])
# input matrix
G=nvect(-.5*deltaT*deltaT, -deltaT)
# observation matrix
H=na([[1, 0]])

""" uncertainty """
n = 2  # number of states
q = 0.5  # std of process
r = 2  # std of measurement
# process noise covariance
Q = q ** 2 * np.eye(n)  # covariance of process
#measurement noise covar.
R = r ** 2  # covariance of measurement

#Q=np.zeros(2)
#R=na([4])

# true init state
x_0=nvect(100, 0)
# assumed init state
x_0_hat=nvect(105, 0)
# assumed init state err covar matrix
P=na([[10, 0],[0, 0.01]])


""" process and measurement simulation """
sim_dur=2#s
sim_len=int(sim_dur/deltaT)
x_k=np.zeros((2,sim_len))
x_k[:,0]=x_0.flatten()

y_k=np.zeros((1,sim_len))

for k in range(1, sim_len):
    x_k[:, k:k + 1]= F @ x_k[:, k - 1:k] + G * u #+ q * randn(2, 1)
    
y_k=H @ x_k + np.random.randn(y_k.size)*np.sqrt(R) # /2 may come from different behavour of randn in py
    

""" Kalman filter """
x_k_hat=np.zeros((2,sim_len))
x_k_hat[:,0]=x_0_hat.flatten()

Kk=np.zeros((2,sim_len))

for k in range(1, sim_len):
    # prediction disregarding new sensor data
    x_kk1 = F @ x_k_hat[:,k-1:k] + G * u
    # state covariance prediction
    P_kk1 = F @ P @ F.T + Q
    
    # Kalman gain
    K = P_kk1 @ H.T @ np.linalg.inv(H @ P_kk1 @ H.T + R)
    Kk[:,k]=K.flatten()
    
    # estimate of the state is weighted sum of prediction and measurement
    x_k_hat[:,k:k+1] = x_kk1 + K @ (y_k[:,k:k+1] - H @ x_kk1)
    # update state covariance
    P = (np.eye(2) - K @ H) @ P_kk1


""" UKF """
def f(x):
    return F @ x + G * u


def h(x):
    return H @ x  # measurement equation


x_hat = x_0_hat
P = eye(n)  # initial state covariance
x_k_hatU = zeros((n, sim_len))  # estimate
KkU=np.zeros((2,sim_len))

for k in range(sim_len):
    x_hat, P, KkU[:,k:k+1] = ukf(f, x_hat, P,
                                 h, y_k[0,k], Q, R)  # ukf
    x_k_hatU[:, k] = x_hat.flatten()  # save estimate

""" plot """

t=np.arange(sim_len) *deltaT
fig=plt.gcf()
fig.clf()
ax1=plt.subplot(2,2,1)
plt.plot(t,x_k[0,:], color=cmap(0), label="true")
plt.plot(t,y_k[0,:],'.', color=cmap(10), label="measurements")
plt.plot(t,x_k_hat[0,:], linewidth=2,color=cmap(4), label="estimate")
plt.plot(t,x_k_hatU[0,:],linewidth=2,color=cmap(12), label="UKF")
plt.ylabel("position [m]")
plt.legend()

ax1=plt.subplot(2,2,3)
plt.plot(t,x_k[1,:], color=cmap(0))
plt.plot(t,x_k_hat[1,:],linewidth=2, color=cmap(4))
plt.plot(t,x_k_hatU[1,:],linewidth=2, color=cmap(12), label="UKF")
plt.ylabel("velocity [m/s]")
plt.xlabel("time [s]")

axK=plt.subplot(2,2,2)
plt.plot(t,Kk[0,:],'--', color=cmap(4), label="KF gain 0")
plt.plot(t,Kk[1,:],color=cmap(5), label="KF gain 1")
plt.legend()

axU=plt.subplot(2,2,4)
plt.plot(t,KkU[0,:],'--', color=cmap(12), label="UKF gain 0")
plt.plot(t,KkU[1,:],color=cmap(13), label="UKF gain 1")
plt.legend()
axU.set_ylim(axK.get_ylim())