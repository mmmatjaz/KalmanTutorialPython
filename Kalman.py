# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi, dot, sum, tile, log, exp
from numpy.linalg import inv, det


class Kalman(object):
    @staticmethod
    def _gauss_pdf(X, M, S):
        if M.shape[1] == 1:
            DX = X - tile(M, X.shape[1])
            E = 0.5 * sum(DX * (inv(S) @ DX), axis=0)
            E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
            P = exp(-E)
        elif X.shape[1] == 1:
            DX = tile(X, M.shape[1]) - M
            E = 0.5 * sum(DX * (dot(inv(S) @ DX)), axis=0)
            E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
            P = exp(-E)
        else:
            DX = X - M
            E = 0.5 * DX.T @ inv(S) @ DX
            E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
            P = exp(-E)
        return P[0], E[0]

    @staticmethod
    def predict(X, P, A, Q, B, U):
        """
        State prediction based on previous measurement
        :param X:  mean state estimate of the previous step (k−1)
        :param P:  state covariance of previous step (k−1)
        :param A:  transition n n × matrix
        :param Q:  process noise covariance matrix.
        :param B:  input effect matrix.
        :param U:  control input.
        :return:
        """
        X = A @ X + B @ U
        P = A @ P @ A.T + Q
        return X, P

    @staticmethod
    def update(X_, P_, Y, H, R):
        """

        :param X:
        :param P:
        :param Y:
        :param H:
        :param R:
        :return:
            K : the Kalman Gain matrix
            IM : the Mean of predictive distribution of Y
            IS : the Covariance or predictive mean of Y
            LH : the Predictive probability (likelihood) of measurement which is
            computed using the Python function gauss_pdf.
        """
        IM = H @ X_
        
        V=Y-IM
        S=H @ P_ @ H.T + R
        K=P_ @ H.T @ inv(S)
        X=X_+K @ V
        P=P_ - K @ S @ K.T
        
        """
        K = P @ H.T @ S
        X = X + K @ (Y - IM)
        P = P - K @ S @ K.T
        LH = Kalman._gauss_pdf(Y, IM, S)
        """
        
        return X, P, K, IM, S#, LH

