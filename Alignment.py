from collections import defaultdict

import numpy as np

Input = open('alignment/alignment_example4.input').read().splitlines()

# Initializing variables A, B, local flag, dx, ex, dy, ey
A = Input[0]
B = Input[1]

lx = len(A) + 1
ly = len(B) + 1
eps = 10e-7
local = Input[2] == '1'

dx, ex, dy, ey = map(float, Input[3].split())

_ = Input[4:8]
# P - match matrix - dictionary with key-value pairs, i.e. MM['AA'] = 1, MM['AT'] = 0.2
s = defaultdict()
for row in Input[8:]:
    row = row.split()
    if row:
        s[row[2] + row[3]] = float(row[4])

# Initialize score matrixes with zeros
M = np.zeros((ly, lx))
Ix = np.zeros((ly, lx))
Iy = np.zeros((ly, lx))

# T - path traceback matrix. 010 - best score from top-left corner (j-1, i-1), 001 - from top (j-1, i),
# 100 - from left (j, i-1), 110 - left and accross, 011 - accross and top, 111 - from every direction
T = np.zeros((ly, lx), dtype=int)

# Preset values of boundary cases for global alignment - top row and left column
# NOT DOING this because of the requirement of not penalizing end-gaps
'''if not local:
    for i in range(1, lx):
        V[0, i] = -dx-ex*(i-1)

    for j in range(1, ly):
        V[j, 0] = -dy-ey*(i-1)'''

# The score matrix reccurence procedure, according to (2.16)
# Left to right, column by column, row by row, we calculate V, X and Y values of the score matrices given their left, across and top neighbors
for i in range(1, lx):
    for j in range(1, ly):
        # find maximum and fill traceback
        m = M[j - 1, i - 1] + s[A[i - 1] + B[j - 1]]
        ix = Ix[j - 1, i - 1] + s[A[i - 1] + B[j - 1]]
        iy = Iy[j - 1, i - 1] + s[A[i - 1] + B[j - 1]]
        if local:
            M[j, i] = max(m, ix, iy, 0)
        else:
            M[j, i] = max(m, ix, iy)

        # we don't trace X and Y, so just set equal to max
        if local:
            Ix[j, i] = max(M[j, i - 1] - dy, Ix[j, i - 1] - ey, 0)
            Iy[j, i] = max(M[j - 1, i] - dx, Ix[j - 1, i] - ex, 0)
        else:  # for global we are not penalizing end-gaps
            if j == ly - 1 or j == 0:
                Ix[j, i] = max(M[j, i - 1], Ix[j, i - 1])
            else:
                Ix[j, i] = max(M[j, i - 1] - dy, Ix[j, i - 1] - ey)
            if i == lx - 1 or i == 0:
                Iy[j, i] = max(M[j - 1, i], Iy[j - 1, i])
            else:
                Iy[j, i] = max(M[j - 1, i] - dx, Iy[j - 1, i] - ex)
        # max values in all 3 score matrices to choose traceback:
        m = M[j, i]
        ix = Ix[j, i]
        iy = Iy[j, i]
        max_val = max(m, ix, iy)
        # if local and less or EQUAL to 0, don't do traceback.
        # This will cut the previous path, even if current score itself was 0. Is it the desired behaviour?
        if not local or max_val > eps:  # the are some rounding errors in the score matrix math, i.e. value 10e-17
            if abs(m - max_val) < eps:
                T[j, i] += 10
            if abs(ix - max_val) < eps:  # this damn rounded float comparison bug has cost me 2 hours alone
                T[j, i] += 100
            if abs(iy - max_val) < eps:
                T[j, i] += 1

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


# finding best score locations
best_loc = []
if local:
    m = M.max()
    ix = Ix.max()
    iy = Iy.max()
else:  # for global we are looking for the best score in all cells of last row and last column
    m = max(M[:-1].max(), M[-1:].max())
    ix = max(Ix[:-1].max(), Ix[-1:].max())
    iy = max(Iy[:-1].max(), Iy[-1:].max())

max_val = max(m, ix, iy)
if abs(m - max_val) < eps:
    best_loc += np.argwhere(M == m).tolist()
if abs(ix - max_val) < eps:
    best_loc += np.argwhere(Ix == ix).tolist()
if abs(iy - max_val) < eps:
    best_loc += np.argwhere(Iy == iy).tolist()

best_loc = [x for x in set(tuple(x) for x in best_loc)]

f = open('output.txt', 'w')
f.write(str(max_val) + '\n\n')
print 'Best score: ', max_val, '\n'
# for every best score find all A-sequences and B-sequences
for best in best_loc:
    rec_traceback('', '', best)

for i in range(len(Apaths)):
    f.write(Apaths[i][::-1] + '\n' + Bpaths[i][::-1] + '\n\n')
    print Apaths[i][::-1]
    print Bpaths[i][::-1], '\n'
# save score matrices for analysis or debug purposes
np.savetxt("V.csv", M, delimiter=",")
np.savetxt("X.csv", Ix, delimiter=",")
np.savetxt("Y.csv", Iy, delimiter=",")
np.savetxt("T.csv", T, delimiter=",")
