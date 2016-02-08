__author__ = 'alex'

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from numpy.random import choice

Input = open('alignment/alignment_example6.input').read().splitlines()

# Initializing variables A, B, local flag
A = Input[0]
B = Input[1]

lx = len(A) + 1
ly = len(B) + 1
eps = 10e-9
ninf = -float('inf')
local = Input[3] == '1'

# Global and local HMM parameters
k = 1  # coefficient to try different sequence total_len, preserving the ratios
tau = 0.005
delta = 0.3
epsilon = 0.7  # 0.8 to be equal 1-2*delta-tau
eta = 0.05

P = defaultdict(lambda: 0)
# P_xy
P['AA'] = 0.175
P['AT'] = 0.025
P['AG'] = 0.025
P['AC'] = 0.025

P['TA'] = 0.025
P['TT'] = 0.175
P['TG'] = 0.025
P['TC'] = 0.025

P['GA'] = 0.025
P['GT'] = 0.025
P['GG'] = 0.175
P['GC'] = 0.025

P['CA'] = 0.025
P['CT'] = 0.025
P['CG'] = 0.025
P['CC'] = 0.175

# dictionary key inverses
for key in P:
    P[key[::-1]] = P[key]

# P_x == P_y
Q = defaultdict(lambda: 0)
Q['A'] = 0.25
Q['T'] = 0.25
Q['G'] = 0.25
Q['C'] = 0.25

# log-odds terms
buf_odd = log((1 - 2 * delta - tau) / (1 - eta) ** 2)
dim = 3


def s(a, b):
    return log(P[a + b] / Q[a] / Q[b]) + buf_odd


d = -log(delta * (1 - epsilon - tau) / (1 - eta) / (1 - 2 * delta - tau))
e = -log(epsilon / (1 - tau))


def Viterbi(A, B):
    # Initialize score matrixes for all 7 states (M,X,Y,RX1,RX2,RY1,RY2) with zeros
    # V = np.zeros((7, ly, lx))
    V = np.zeros((dim, ly, lx))  # initialize separately for local case

    # T - path traceback matrix. 010 - best score from top-left corner (j-1, i-1), 001 - from top (j-1, i),
    # 100 - from left (j, i-1), 110 - left and accross, 011 - accross and top, 111 - from every direction
    T = np.zeros((ly, lx), dtype=int)

    for i in range(1, lx):
        for j in range(1, ly):
            if not local:
                # V[k,j,i]
                V[1:, 1, 1] = ninf
                V[:, 0, :] = V[:, :, 0] = ninf
                if i == 1 and j == 1:
                    V[0, 1, 1] = log(tau) - 2 * log(eta)
                else:
                    V[0, j, i] = s(A[i - 1], B[j - 1]) + max(V[:, j - 1, i - 1])
                    V[1, j, i] = max(V[0, j, i - 1] - d, V[1, j, i - 1] - e)
                    V[2, j, i] = max(V[0, j - 1, i] - d, V[2, j - 1, i] - e)
                max_val = max(V[:, j, i])
            else:
                V = np.zeros((dim, ly, lx))
                V[0, 0, 0] = 1
                V[:, 0, :] = V[:, :, 0] = 0

                for i in range(1, lx):
                    for j in range(1, ly):
                        rij = random_model(i - 1, j - 1)
                        rij2 = random_model(i, j - 1)
                        V[0, j, i] = P[A[i] + B[j]] * max(rij * (1 - 2 * delta - tau),
                                                          (1 - 2 * delta - tau) * V[0, j - 1, i - 1],
                                                          (1 - epsilon - tau) * V[1, j - 1, i - 1], V[2, j - 1, i - 1])
                        V[1, j, i] = Q[A[i]] * max(rij2 * delta, delta * V[0, j, i - 1], epsilon * V[1, j, i - 1])
                        V[2, j, i] = Q[B[j]] * max(rij2 * delta, delta * V[0, j - 1, i], epsilon * V[2, j - 1, i])
                max_val = max(V[:, j, i])
                if rij > max_val:
                    T = 10
                elif rij2 > max_val:
                    T = 101

            if abs(V[0, j, i] - max_val) < eps:
                T[j, i] += 10
            if abs(V[1, j, i] - max_val) < eps:
                T[j, i] += 100
            if abs(V[2, j, i] - max_val) < eps:
                T[j, i] += 1

    c = log(1 - 2 * delta - tau) - log(1 - epsilon - tau)
    max_val = max(V[0, ly - 1, lx - 1], V[1, ly - 1, lx - 1] + c, V[2, ly - 1, lx - 1] + c)

    # Finding the paths which lead to best score
    # A, Bs - lists of all optimal A and B sequences
    Apaths = []
    Bpaths = []

    # recursively build all possible paths step by step, splitting into 2 or 3 paths in cases 11, 110 and 111 (multiple best score sources)
    def rec_traceback(a, b, best_loc):
        j, i = best_loc
        dir = T[j, i]
        if not dir:
            Apaths.append(a)
            Bpaths.append(b)
            return

        if dir == 111:
            rec_traceback(a + A[i - 1], b + B[j - 1], [j - 1, i - 1])
            rec_traceback(a + A[i - 1], b + '_', [j, i - 1])
            rec_traceback(a + '_', b + B[j - 1], [j - 1, i])

        elif dir == 110:
            rec_traceback(a + A[i - 1], b + B[j - 1], [j - 1, i - 1])
            rec_traceback(a + A[i - 1], b + '_', [j, i - 1])
        elif dir == 11:
            rec_traceback(a + A[i - 1], b + B[j - 1], [j - 1, i - 1])
            rec_traceback(a + '_', b + B[j - 1], [j - 1, i])
        elif dir == 101:
            rec_traceback(a + A[i - 1], b + '_', [j, i - 1])
            rec_traceback(a + '_', b + B[j - 1], [j - 1, i])
        elif dir == 100:
            rec_traceback(a + A[i - 1], b + '_', [j, i - 1])
        elif dir == 10:
            rec_traceback(a + A[i - 1], b + B[j - 1], [j - 1, i - 1])
        elif dir == 1:
            rec_traceback(a + '_', b + B[j - 1], [j - 1, i])

    print 'Best score (Viterbi algorithm):\n', max_val, '\n'
    rec_traceback('', '', [ly - 1, lx - 1])
    print 'Best alignments:'
    for i in range(len(Apaths)):
        print Apaths[i][::-1]
        print Bpaths[i][::-1], '\n'


def fwd(A, B):
    F = np.zeros((dim, ly, lx))
    F[1:, 1, 1] = 0
    F[:, 0, :] = F[:, :, 0] = 0

    for i in range(1, lx):
        for j in range(1, ly):
            if i == 1 and j == 1:
                F[0, 1, 1] = 1
            else:
                F[0, j, i] = P[A[i - 1] + B[j - 1]] * (
                    (1 - 2 * delta - tau) * F[0, j - 1, i - 1] + (1 - epsilon - tau) * (
                        F[1, j - 1, i - 1] + F[2, j - 1, i - 1]))
                F[1, j, i] = Q[A[i - 1]] * (delta * F[0, j, i - 1] + epsilon * F[1, j, i - 1])
                F[2, j, i] = Q[B[j - 1]] * (delta * F[0, j - 1, i] + epsilon * F[2, j - 1, i])
    return F


def bkw(A, B):
    Bw = np.zeros((dim, ly, lx))
    Bw[:, -2, -2] = tau
    Bw[:, -1, :] = Bw[:, :, -1] = 0
    A += ' '
    B += ' '
    for i in range(lx - 2, -1, -1):
        for j in range(ly - 2, -1, -1):
            if i == lx - 2 and j == ly - 2:
                continue
            Bw[0, j, i] = (1 - 2 * delta - tau) * P[A[i + 1] + B[j + 1]] * Bw[0, j + 1, i + 1] + delta * (
                Q[A[i + 1]] * Bw[1, j, i + 1] + Q[B[j + 1]] * Bw[2, j + 1, i])
            Bw[1, j, i] = (1 - epsilon - tau) * P[A[i + 1] + B[j + 1]] * Bw[0, j + 1, i + 1] + epsilon * Q[A[i + 1]] * \
                                                                                               Bw[1, j, i + 1]
            Bw[2, j, i] = (1 - epsilon - tau) * P[A[i + 1] + B[j + 1]] * Bw[0, j + 1, i + 1] + epsilon * Q[B[j + 1]] * \
                                                                                               Bw[2, j + 1, i]
    A = A.strip()
    B = B.strip()
    return Bw


def random_model(lx, ly):
    return log(eta ** 2 * (1 - eta) ** (lx + ly - 2)) + log(Q['A']) * lx + log(Q['A']) * ly


def Forward(A, B):
    F = fwd(A, B)
    # the null model probability
    null_prob = random_model(lx, ly)
    print 'Overall ll of sequences being related opposed to random model (Forward algorithm):'
    print log(tau * sum(F[:, ly - 1, lx - 1])) / null_prob


def Forward_Backward(A, B):
    Fw = fwd(A, B)
    Bw = bkw(A, B)
    return Fw, Bw


def FB_map(A, B):
    Fw, Bw = Forward_Backward(A, B)
    hm = np.zeros((len(B), len(A)))
    for j in range(len(B)):
        for i in range(len(A)):
            hm[j, i] = log(max(Fw[:, j + 1, i + 1]) * max(Bw[:, j, i])) / log(tau * sum(Fw[:, ly - 1, lx - 1]))
    hm[hm == np.inf] = 0.1
    hm = hm / hm.max()
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.set_title('Residue alignment probability')
    plt.imshow(hm)
    ax.set_aspect('equal')
    ax.set_xlabel(A)
    ax.set_ylabel(B)
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()


def Probabilistic_Sampling(A, B):
    newA = ''
    newB = ''
    F = fwd(A, B)
    j, i = ly, lx
    cur_state = 0
    while j > 0 and i > 0:
        if cur_state == 0:
            if i < len(A) and j < len(B):
                newA += A[i - 1]
                newB += B[j - 1]
            m = [(1 - 2 * delta - tau) * F[0, j - 1, i - 1], (1 - epsilon - tau) * F[1, j - 1, i - 1],
                 (1 - epsilon - tau) * F[2, j - 1, i - 1]]
            cur_state = choice([0, 1, 2], p=m / sum(m))
            j -= 1
            i -= 1
        elif cur_state == 1:
            newA += A[i - 1]
            m = [delta * F[0, j, i - 1], tau * F[1, j, i - 1]]
            cur_state = choice([0, 1], p=m / sum(m))
            i -= 1
        else:  # cur_state == 2
            newB += B[j - 1]
            m = [delta * F[0, j - 1, i], tau * F[1, j - 1, i]]
            cur_state = choice([0, 2], p=m / sum(m))
            j -= 1
    return newA, newB


Viterbi(A, B)
Forward(A, B)
print '\n\nProbabilistic sampling:\n'
for i in range(10):
    buf_old = set()
    newA, newB = Probabilistic_Sampling(A, B)
    if not newA + newB in buf_old:
        buf_old.add(newA + newB)
        print newA[::-1]
        print newB[::-1] + '\n'

# print 'Posterior probability of residues being aligned (Forward-Backward algorithm)\n'
FB_map(A, B)
