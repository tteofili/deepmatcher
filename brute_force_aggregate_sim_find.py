# -*- coding: utf-8 -*-
"""
"""

from __future__ import print_function

from inspect import getsource


import numpy as np
import sim_function
from csv2dataset import csv_2_datasetALTERNATE

simfunctions = [
    lambda t1, t2: sim_function.sim_jacc(t1, t2),
    lambda t1, t2: sim_function.sim_cos(t1, t2),
    lambda t1, t2: sim_function.sim_hamming(t1, t2),
    lambda t1, t2: sim_function.sim_lcs(t1, t2),
    lambda t1, t2: sim_function.sim_lev(t1, t2),
    lambda t1, t2: sim_function.sim_ngram(t1, t2),
    lambda t1, t2: sim_function.sim_sodi(t1, t2),
]

get_lambda_name = lambda l: getsource(l).split('=')[0].strip()

DATASET_NAME = 'dplb_scholar'
GROUND_TRUTH_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/DBLP-Scholar_perfectMapping.csv'
TABLE1_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/DBLP1.csv'
TABLE2_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/Scholar.csv'
ATT_INDEXES = [(1, 1), (2, 2), (3, 3), (4,4)]

runs = 1

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
            #print(f'checking {k}:{name}')
            data = csv_2_datasetALTERNATE(gt_file, t1_file, t2_file, [k], simf)
            perc = len(data) * 0.05
            npdata = np.array(data[:int(perc)])
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
            #print(f'{k}:{name}:{mse}')
            if mse < lowestMSE:
                lowestMSE = mse
                bestFun = name
                #print(f'update for {k}: function={bestFun}, MSE={lowestMSE}')
        print(f'BEST for {k}: function={bestFun}, MSE={lowestMSE}')
        best.append(bestFun)
    print(f'final aggregated function: {best}')
    return best


bestfuns = []
for i in range(runs):
    bestfuns.append(bf(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simfunctions))
print(bestfuns)