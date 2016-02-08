import math
import time
from collections import OrderedDict

import numpy as np
from ete2 import Tree

Input = open('simple_tests/sample0.nex').read().splitlines()
alpha = 0.5  # 0.402359478 # since Transition/transversion ratio = 2.0 in the simple tests.

# Jukes & Cantor parameters:
Rt = {}  # dictionary with r_t values
St = {}  # dictionary with s_t values


# computes P(b|a,t)
def PL(a, b, t):
    if b == a:
        if not t in Rt:
            Rt[t] = (1 + 3 * math.e ** (-4 * alpha * t)) / 4
        return Rt[t]
    else:
        if not t in S:
            St[t] = (1 - math.e ** (-4 * alpha * t)) / 4
        return St[t]


S = OrderedDict()  # dictionary of sequences
T = []  # list of trees topologies
q = {}  # dictionary of frequencies

for line in Input:
    if '\'' in line:
        line = line.split(' ')
        S[line[0].strip('\'')] = line[-1]
    elif '=' in line:
        T.append(line.split('=')[-1].strip())

s_all = ''.join(S.values())  # join all sequences to determine the frequencies
total_len = float(len(s_all))
q['A'] = s_all.count('A') / total_len if s_all.count(
    'A') else 0  # in case of small input, where some residue don't exist
q['C'] = s_all.count('C') / total_len if s_all.count('C') else 0
q['G'] = s_all.count('G') / total_len if s_all.count('G') else 0
q['T'] = s_all.count('T') / total_len if s_all.count('T') else 0


# computes log likelihood for a single site
def Felsenstein(root, site, R):
    def recursion(node, residue):
        if node.is_leaf():
            return residue == S[node.name][site]

        leaves = ''.join([S[l.name][site] for l in node.get_leaves()])
        if (node, residue, leaves) in R:
            return R[node, residue, leaves]

        if len(node.children) > 2:
            child1, child2, child3 = node.children
            return sum([PL(residue, b, child1.dist) * recursion(child1,
                                                                b) *  # test if adding 0 length edge for true binarization would help
                        PL(residue, c, child2.dist) * recursion(child2, c) *
                        PL(residue, d, child3.dist) * recursion(child3, d)
                        for b in ['A', 'C', 'G', 'T'] for c in ['A', 'C', 'G', 'T'] for d in ['A', 'C', 'G', 'T']])
        else:
            child1, child2 = node.children
            R[node, residue, leaves] = \
                sum([PL(residue, b, child1.dist) * recursion(child1, b) *
                     PL(residue, c, child2.dist) * recursion(child2, c)
                     for b in ['A', 'C', 'G', 'T'] for c in ['A', 'C', 'G', 'T']])
            return R[node, residue, leaves]

    return math.log(sum([recursion(root, a) * q[a] for a in ['A', 'C', 'G', 'T']]))


# finds the tree with the best log likelihood
def find_ll():
    best_tree = 0
    max_ll = -np.inf
    # for every tree
    for i, t in enumerate(T):
        root = Tree(t)
        ll = 0
        old_sites = {}  # dictionary to avoid repetitive calculations of exactly the same sites
        R = {}  # dictionary for memorizing recursion steps to avoid repetition
        # for every site
        for site in range(len(S.values()[0])):
            siteTrace = ''.join([s[site] for s in S.values()])
            if not siteTrace in old_sites:
                old_sites[siteTrace] = Felsenstein(root, site, R)
            # add site likelihood to the total log likelihood
            ll += old_sites[siteTrace]
        if ll > max_ll:
            # choose the best tree
            max_ll = ll
            best_tree = i
    print max_ll
    print T[best_tree]
    with open('output.txt', 'w') as f:
        f.write(T[best_tree] + '\n\n' + 'log likelihood: ' + str(max_ll))


t = time.time()
find_ll()
print 'duration: ', time.time() - t
