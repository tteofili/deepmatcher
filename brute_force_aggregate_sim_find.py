# -*- coding: utf-8 -*-
"""
"""

from __future__ import print_function

from inspect import getsource


import numpy as np
import sim_function
from csv2dataset import read_data_bf
from csv2dataset import csv_2_datasetALTERNATE
from sklearn import linear_model
import operator


simfunctions = [
    lambda t1, t2: sim_function.sim_noselect(t1, t2),
    lambda t1, t2: sim_function.sim_equals(t1, t2),
    lambda t1, t2: sim_function.jaro(t1, t2),
    lambda t1, t2: sim_function.sim_jacc(t1, t2),
    lambda t1, t2: sim_function.sim_cos(t1, t2),
    lambda t1, t2: sim_function.sim_hamming(t1, t2),
    lambda t1, t2: sim_function.sim_lcs(t1, t2),
    lambda t1, t2: sim_function.sim_lev(t1, t2),
    lambda t1, t2: sim_function.sim_ngram(t1, t2),
    lambda t1, t2: sim_function.sim_sodi(t1, t2),
    lambda t1, t2: sim_function.sim_bert(t1, t2),
    lambda t1, t2: sim_function.sim_sbert(t1, t2),
    lambda t1, t2: sim_function.sim_sbert2(t1, t2),
    lambda t1, t2: sim_function.sim_sodi(t1.split(), t2.split()),
    lambda t1, t2: sim_function.jaro(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_cos(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_hamming(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_lcs(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_lev(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_ngram(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_bert(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_sbert(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_sbert2(t1.split(), t2.split()),
]


get_lambda_name = lambda l: getsource(l).split('=')[0].strip()

DATASET_NAME = 'abt_buy_anhai'
GROUND_TRUTH_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/all.csv'
TABLE1_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/tableA.csv'
TABLE2_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/tableB.csv'
ATT_INDEXES = [(1, 1), (2, 2), (3, 3)]
'''
DATASET_NAME = 'dplb_scholar'
GROUND_TRUTH_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/DBLP-Scholar_perfectMapping.csv'
TABLE1_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/DBLP1.csv'
TABLE2_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/Scholar.csv'
ATT_INDEXES = [(1, 1), (2, 2), (3, 3), (4, 4)]
'''

runs = 3

def unflat(data):
    u_data = np.zeros_like(data)
    i = 0
    for d in data:
        u_data[i] = d[0]
        i += 1
    return u_data


def bf(gt_file, t1_file, t2_file, attr_indexes, sim_functions):
    best = []
    for k in attr_indexes:
        bestFun = lambda t1, t2: t1 == t2
        lowestMSE = 1e10
        for simf in sim_functions:
            name = get_lambda_name(simf)
            data = csv_2_datasetALTERNATE(gt_file, t1_file, t2_file, [k], sim_functions[4])
            #data = read_data_bf(gt_file, t1_file, t2_file, [k], simfunctions,  simf)
            perc = len(data) * 0.05
            split = int(max(perc / 2, 20))
            npdata = np.array(data[:split] + data[-split:])
            npdata[:, 2] = unflat(npdata[:, 2])
            ones = npdata[np.where(npdata[:, 3] == 1)][:, 3]
            ones_sim = npdata[np.where(npdata[:, 3] == 1)][:, 2]
            zeros = npdata[np.where(npdata[:, 3] == 0)][:, 3]
            zeros_sim = npdata[np.where(npdata[:, 3] == 0)][:, 2]
            mse_ones = np.square((ones - ones_sim)).mean(axis=None)
            mse_zeros = 0
            if len(zeros_sim > 0):
                mse_zeros = np.square((zeros - zeros_sim)).mean(axis=None)
            alpha = 0.5 + (len(ones_sim) - len(zeros_sim)) / (len(ones_sim) + len(zeros_sim))
            mse = mse_ones * alpha + mse_zeros * (1 - alpha)
            #mse = np.square((npdata[:, 2] - npdata[:, 3])).mean(axis=None)
            print(f'{k}:{name}:{mse}')
            if mse < lowestMSE:
                lowestMSE = mse
                bestFun = simf
                #print(f'update for {k}: function={bestFun}, MSE={lowestMSE}')

        print(f'BEST for {k}: function={get_lambda_name(bestFun)}, MSE={lowestMSE}')
        best.append(bestFun)
    print(f'final aggregated function: {best}')
    return best

def bf2(gt_file, t1_file, t2_file, attr_indexes, sim_functions):
    best = []
    for k in attr_indexes:
        print('getting attribute values')
        data = csv_2_datasetALTERNATE(gt_file, t1_file, t2_file, [k], sim_functions[2])
        perc = len(data) * 0.05
        split = int(max(perc/2, 50))
        npdata = np.array(data[:split] + data[-split:])
        X = np.zeros([len(npdata), len(sim_functions)])
        Y = np.zeros(len(npdata))
        tc = 0
        print('building training set')
        for t in npdata:
            ar = np.zeros(len(sim_functions))
            arc = 0
            for s in sim_functions:
                ar[arc] = np.nan_to_num(s(t[0][0], t[1][0])[0])
                arc += 1
            X[tc] = ar
            Y[tc] = t[3]
            tc += 1
        print('fitting classifier')
        clf = linear_model.LinearRegression()
        clf.fit(X, Y)
        print(f'score: {clf.score(X,Y)}')
        weights = clf.coef_
        comb = []
        combprint = []
        normalized_weights = weights
        if min(weights) < 0:
            normalized_weights = normalized_weights + abs(min(weights))
        wsum = np.sum(normalized_weights)
        '''
         for c in range(len(weights)):
            comb.append([sim_functions[c], normalized_weights[c]/wsum])
            combprint.append([get_lambda_name(sim_functions[c]), normalized_weights[c]/wsum])
        '''
        '''for c in range(len(weights)):
            comb.append([sim_functions[c], weights[c]])
            combprint.append([get_lambda_name(sim_functions[c]), weights[c]])'''
        comb.sort(key=operator.itemgetter(1), reverse=True)
        combprint.sort(key=operator.itemgetter(1), reverse=True)

        print(f'sim weights for {k}: {combprint}')

        best.append(comb)
    return best


'''print(f'finding bf function for {DATASET_NAME}')
bestfuns = []
for i in range(runs):
    bestfuns.append(bf2(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simfunctions))
print(bestfuns)
'''