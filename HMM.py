__author__ = 'alex'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import choice

# map letters to state numbers
vocab = {0: 'A', 1: 'C', 2: 'G', 3: 'T',
         'A': 0, 'C': 1, 'G': 2, 'T': 3}

# input parameters for sequnce simulation
n = 1000
eps1 = 0.02  # 1/(0.000000155 * len(X))# 465/3*10e9                      0.002150538 #1/465 # "+" -> "-" transition probability
eps2 = 0.005  # 0.000009667  0.004495

ab = 0.0001
ae = 0.000001

an = np.array(
    [[0.3, 0.205, 0.285, 0.21],
     [0.322, 0.298, 0.078, 0.302],
     [0.248, 0.246, 0.298, 0.208],
     [0.177, 0.239, 0.292, 0.292]])

ap = np.array(
    [[0.18, 0.274, 0.426, 0.12],
     [0.171, 0.368, 0.274, 0.188],
     [0.161, 0.339, 0.375, 0.125],
     [0.079, 0.355, 0.384, 0.182]])

#           n=>n        n=>p            p=>n        p=>p
# A = np.concatenate([(1-eps2) * an, eps2 * an, eps1 * ap, (1-eps1) * ap], axis=1)
#                       n=>np                                       p=>np
A_original = np.array(
    [np.concatenate([(1 - eps2) * an, eps2 * an], axis=1), np.concatenate([eps1 * ap, (1 - eps1) * ap], axis=1)])
Ab = [0.132205099363, 0.126700662417, 0.122638045244, 0.127895013123, 0.0725193100862, 0.163763029621, 0.178824271966,
      0.075454568179]
Ae = eps2 / 10
# number of possible transitions from one state
m = A_original.shape[2]


# def transmition(state, obs):

def simulate_sequence():
    X = ''
    S = ''
    p = [A_original[0, :, i].sum() for i in range(m)]
    obs = choice(m, p=p / sum(p))

    for i in range(n):
        if obs < m / 2:
            state = 0
        else:
            obs %= m / 2
            state = 1
        X += vocab[obs]
        S += '+' if state else '-'
        p = A_original[state, obs, :]
        obs = choice(m, p=p / sum(p))
    return X, S


Xs = []
Ss = []

for i in range(10):
    s = simulate_sequence()
    Xs.append(s[0])
    Ss.append(s[1])
    # print P[0]
    # print P[1]
    if i == 0:
        pd.DataFrame([1 if a == '+' else 0 for a in list(s[1])]).plot(figsize=(20, 1), kind='area', title='Real labels')


def fwd(X, A):
    # initialize first state
    obs_prev = vocab[X[0]]

    F = np.zeros((2, len(X)))
    F[0, 0] = A[0, :, obs_prev].sum() / A[0].sum()
    F[1, 0] = A[1, :, obs_prev + m / 2].sum() / A[1].sum()
    sf = [np.log(1 / sum(F[:, 0]))]

    for i in range(1, len(X)):
        obs_cur = vocab[X[i]]
        for v in range(len(F)):
            F[v, i] = sum([F[k, i - 1] * A[k, obs_prev, obs_cur + v * m / len(F)] for k in range(len(F))])
        sf += [sf[-1] + np.log(1 / sum(F[:, i]))]
        F[:, i] /= sum(F[:, i])  # normalization preventing underflow
        obs_prev = obs_cur
    return np.log(F) - sf


def bkw(X, A):
    # initialize first state
    obs_next = vocab[X[-1]]

    B = np.zeros((2, len(X)))
    B[0, -1] = B[1, -1] = 0.5  # 0.0001 will change results of forward and backward scores
    sb = [np.log(1 / sum(B[:, -1]))]

    for i in range(len(X) - 2, -1, -1):
        obs_cur = vocab[X[i]]
        for v in range(len(B)):
            B[v, i] = sum([B[k, i + 1] * A[k, obs_cur, obs_next + v * m / len(B)] for k in range(len(B))])
        sb += [sb[-1] + np.log(1 / sum(B[:, i]))]
        B[:, i] /= sum(B[:, i])  # normalization preventing underflow
        obs_next = obs_cur
    return np.log(B) - sb[::-1]


def Forward(X, A):
    F = fwd(X, A)
    return np.max(F[:, -1])


def Forward_Backward(X, A):
    F = fwd(X, A)
    B = bkw(X, A)
    P = np.zeros((2, F.shape[1]))
    Px = np.max(F[:, -1])

    for i in range(P.shape[1]):
        for v in range(P.shape[0]):
            P[v, i] = F[v, i] * B[v, i]
    return P / Px


def FB_decoding(X, A):
    f = Forward_Backward(X, A)
    return [np.argmax(f[:, i]) for i in range(f.shape[1])], [f[0, i] / sum(f[:, i]) for i in range(f.shape[1])]


def Viterbi(X, A):
    P = np.zeros((2, len(X)), dtype=int)
    P[1, 0] = 1
    # initialize first state
    obs_prev = vocab[X[0]]
    V = np.zeros((2, len(X)))
    V[0, 0] = np.log10(A[0, :, obs_prev].sum() / A[0].sum())
    V[1, 0] = np.log10(A[1, :, obs_prev + m / 2].sum() / A[1].sum())
    a = np.log10(A)

    for i in range(1, len(X)):
        obs_cur = vocab[X[i]]
        for v in range(len(V)):
            V[v, i] = max([V[k, i - 1] + a[k, obs_prev, obs_cur + v * m / len(V)] for k in range(len(V))])
            P[v, i] = np.argmax([V[k, i - 1] + a[k, obs_prev, obs_cur + v * m / len(V)] for k in range(len(V))])
        obs_prev = obs_cur

    def traceback():
        p = ''
        state = np.argmax(V[:, -1])
        for i in range(P.shape[1] - 1, -1, -1):
            p += str(state)
            state = P[state, i]
        return p[::-1].replace('0', '-').replace('1', '+')

    return traceback()


def Baum_Welch(maxiter=1000):
    A = np.random.rand(A_original.shape[0], A_original.shape[1], A_original.shape[2])
    # ll = Forward(Xs[0],A)
    for _ in xrange(maxiter):
        A_old = np.array(A)
        dA = np.zeros(A.shape)
        for X in Xs:
            F = fwd(X, A)
            B = bkw(X, A)
            ddA = np.zeros(A.shape)
            for i in range(len(X) - 1):
                k = vocab[X[i]]
                l = vocab[X[i + 1]]
                ddA[0, k, l] += F[0, i] * B[0, i + 1] * A[0, k, l]
                ddA[0, k, l + 4] += F[0, i] * B[1, i + 1] * A[0, k, l + 4]
                ddA[1, k, l] += F[1, i] * B[0, i + 1] * A[1, k, l]
                ddA[1, k, l + 4] += F[1, i] * B[1, i + 1] * A[1, k, l + 4]
            dA += ddA / Forward(X, A)
        for v in range(2):
            for k in range(4):
                for l in range(8):
                    A[v, k, l] = dA[v, k, l] / sum(dA[v, k, :])
        # ll_new = Forward(Xs[0],A)
        # if abs(ll - ll_new) > 0.00001:
        #    ll = ll_new
        # else:
        if abs(np.linalg.norm(A - A_old)) < 0.0001:
            return A


print '\nOriginal observations and state sequences:'
print Xs[0]
print Ss[0]
path = Viterbi(Xs[0], A_original)
print '\nViterbi decoding'
print path
pd.DataFrame([1 if a == '+' else 0 for a in path]).plot(figsize=(20, 1), kind='area', title='Viterbi labels')
exit()
old, new = FB_decoding(Xs[0], A_original)

path = pd.DataFrame(old)
path.plot(figsize=(20, 1), kind='area', title='Posterior labels')

path = pd.DataFrame(new)
path.plot(figsize=(20, 1), kind='line', title='Posterior probabilities', ylim=(0.495, 0.505))

# print Baum_Welch()

plt.show()

path = Viterbi(Xs[0], Baum_Welch())
pd.DataFrame([1 if a == '+' else 0 for a in path]).plot(figsize=(20, 1), kind='area',
                                                        title='Viterbi labels on Baum Welch estimated parameters')
