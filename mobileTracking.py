# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:29:55 2018

@author: Matjaz
"""
from matplotlib import gridspec
from numpy import array as na
import numpy as np
import matplotlib.pyplot as plt


def nvect(*argv):
    return np.array([argv]).T


cmap = plt.get_cmap("tab20c")

""" process model """
u = 2.5  

def getU(t):
    if t<5:
        return u
    elif 20<t<22:
        return -u/2
    else: return 0

deltaT = 1
D=100
M=1e4 # 1 ton

# system state
F = na([[1, deltaT], [0, M/(M+deltaT*D)]])
# input matrix
G = nvect(.5 * deltaT * deltaT, deltaT)
# observation matrix
H = na([[1, 0]])
# process noise covariance
Q = np.eye(2)*.1
# measurement noise covar.
R = na([100])

# true init state
x_0 = nvect(0, 0)
# assumed init state
x_0_hat = nvect(50, 0)
# assumed init state err covar matrix
P = na([[1000, 0], [0, 1]])

""" process and measurement simulation """
sim_dur = 60  # s
sim_len = int(sim_dur / deltaT)
x_k = np.zeros((2, sim_len))
x_k[:, 0] = x_0.flatten()

y_k = np.zeros((1, sim_len))
u_k = np.zeros((1, sim_len))

for k in range(1, sim_len):
    uk = getU(k/deltaT)
    x_k[:, k:k + 1] = F @ x_k[:, k - 1:k] + G * getU(k*deltaT)
    u_k[0,k]=uk

y_k = H @ x_k + np.random.randn(y_k.size) * np.sqrt(R)  # /2 may come from different behavour of randn in py

""" Kalman filter """
x_k_hat = np.zeros((2, sim_len))
x_k_hat[:, 0] = x_0_hat.flatten()
for k in range(1, sim_len):
    uk = getU(k/deltaT)
    # predition disregarding new sensor data
    x_kk1 = F @ x_k_hat[:, k - 1:k] + G * getU(k*deltaT)
    # state covariance prediction
    P_kk1 = F @ P @ F.T + Q

    # Kalman gain
    K = P_kk1 @ H.T @ np.linalg.inv(H @ P_kk1 @ H.T + R)

    # estimate of the state is weighted sum of prediction and measurement
    x_k_hat[:, k:k + 1] = x_kk1 + K @ (y_k[:, k:k + 1] - H @ x_kk1)
    # update state covariance
    P = (np.eye(2) - K @ H) @ P_kk1

""" plot """
t=np.arange(sim_len) *deltaT

gs = gridspec.GridSpec(3, 1,height_ratios=[1,2,2])

fig = plt.gcf()
fig.clf()
ax1 = plt.subplot(gs[0])
plt.plot(t, u_k.flatten(), '.', color=cmap(0))
plt.ylabel("acc/brake")

ax1 = plt.subplot(gs[1])
plt.plot(t, x_k[0, :], color=cmap(0))
plt.plot(t, y_k[0, :], '.', color=cmap(8))
plt.plot(t, x_k_hat[0, :], '--', color=cmap(4))
plt.ylabel("position [m]")
plt.legend()


ax1 = plt.subplot(gs[2])
plt.plot(t, x_k[1, :], color=cmap(0))
plt.plot(t, x_k_hat[1, :], '--', color=cmap(4))
plt.ylabel("velocity [m/s]")
plt.xlabel("time [s]")
