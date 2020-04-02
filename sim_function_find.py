from __future__ import print_function
from plot import plot_graph, plot_pretrain, plot_dataPT

import random
import re, math
from collections import Counter
import matplotlib.pyplot as plt
from csv2dataset import splitting_dataSet, csvTable2datasetRANDOM, csv_2_datasetALTERNATE, \
    csvTable2datasetRANDOM_likeGold
# from csv2dataset import csvTable2datasetRANDOMCos,splitting_dataSet, csv_2_datasetALTERNATE, csvTable2datasetRANDOM
# from lsh_forest import minHash_lshForest
from minhash_lsh2 import minHash_lsh
from sim_function import min_cos
from random import shuffle
import numpy as np
from dm_train import train_dm
from dm_train import eval_dm
from dm_train import pretrain_dm
from dm_train import finetune_dm
from dm_train import pt_ft_dm_classifier
from dm_train import join
import sim_function
from inspect import getsource

simfunctions = [
    lambda t1, t2: sim_function.sim_jacc(t1, t2),
    lambda t1, t2: sim_function.sim_cos(t1, t2),
    lambda t1, t2: sim_function.sim_hamming(t1, t2),
    lambda t1, t2: sim_function.sim_lcs(t1, t2),
    lambda t1, t2: sim_function.sim_lev(t1, t2),
    lambda t1, t2: sim_function.sim_ngram(t1, t2),
    lambda t1, t2: sim_function.sim_sodi(t1, t2),
    lambda t1, t2: sim_function.sim4attrFZ(t1, t2),
    lambda t1, t2: sim_function.sim4attrFZwoClassEPhone(t1, t2),
    lambda t1, t2: sim_function.sim4attrFZwoClass(t1, t2),
    lambda t1, t2: sim_function.sim4attrWA(t1, t2),
    lambda t1, t2: sim_function.sim4attrGA(t1, t2),
    lambda t1, t2: sim_function.sim4attrScho(t1, t2),
]

DATASET_NAME = 'dplb_scholar'
GROUND_TRUTH_FILE = '/home/tteofili/Downloads/dataset/'+DATASET_NAME+'/DBLP-Scholar_perfectMapping.csv'
TABLE1_FILE = '/home/tteofili/Downloads/dataset/'+DATASET_NAME+'/DBLP1.csv'
TABLE2_FILE = '/home/tteofili/Downloads/dataset/'+DATASET_NAME+'/Scholar.csv'
ATT_INDEXES = [(1, 1), (2, 2), (3, 3),(4,4)]

tot_pt = 2000  # dimensione dataset pre_training
tot_copy = 900 # numero di elementi generati con edit distance
soglia = 0.03  # da aggiungere per discostarsi da min_sim e max_sim ottenuto
get_lambda_name = lambda l: getsource(l).split('=')[0].strip()

best = []

for r in range(5):
    lowestMSE = 1e10
    bestFun = lambda t1, t2: t1 == t2

    # for each sim function
    for simf in simfunctions:
        data = csv_2_datasetALTERNATE(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        min_sim_Match, max_sim_noMatch = plot_graph(data)
        max_sim = soglia + max(min_sim_Match, max_sim_noMatch)
        if max_sim > 0.94:
            max_sim = 0.82

        min_sim = min(min_sim_Match, max_sim_noMatch)  # -soglia

        def unflat(data, target_idx, shrink=False):

            def cut_string(s):
                if len(s) >= 1000:
                    return s[:1000]
                else:
                    return s

            temp = []
            id = 0
            for r in data:
                t1 = r[0]
                t2 = r[1]
                lb = r[target_idx]
                if isinstance(lb, list):
                    lb = lb[0]
                if (shrink):
                    t1 = list(map(cut_string, t1))
                    t2 = list(map(cut_string, t2))
                row = []
                row.append(id)
                row.append(lb)
                for a in t1:
                    row.append(a)
                for a in t2:
                    row.append(a)
                temp.append(row)
                id = id + 1

            return temp

        deepmatcher_data = unflat(data, 3)


        # Split in train, validation and test set.
        def split_training_valid_test(data, SPLIT_FACTOR=0.8):
            bound = int(len(data) * SPLIT_FACTOR)
            train = data[:bound]
            remain = data[bound:]
            bound2 = int(len(remain) * 0.5)
            test = remain[bound2:]
            valid = remain[:bound2]

            return train, valid, test


        # Tutti i successivi addestramenti partiranno dal 100% di train (80% di tutti i dati).
        # Le tuple in valid non verranno mai usate per addestrare ma solo per testare i modelli.
        train, valid, test = split_training_valid_test(deepmatcher_data)

        datapt_hash = minHash_lsh(TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)

        kappa = [int(tot_pt / 2)]  # 100,250,500,1000]
        # kappa=[tot_pt]#100,250,500,1000]
        for i in range(len(kappa)):
            k = kappa[i]
            # Dataset per VinSim.
            vinsim_data = []

            # Porzione di tuple in match da includere nell'addestramento di VinSim.
            TP_FACTOR = 0.05

            # Preleva solo quelle in match con il relativo sim vector.
            # sim_data = unflat(data, 3)
            for i in range(len(data)):
                if data[i][3] == 1:
                    r = data[i]
                    vinsim_data.append(r)

            # Taglio della porzione desiderata.
            # bound = int(len(vinsim_data) * TP_FACTOR)
            bound = 5
            vinsim_data = vinsim_data[:bound]

            min_cos_sim = min_cos(vinsim_data)

            random_tuples0 = csvTable2datasetRANDOM_likeGold(TABLE1_FILE, TABLE2_FILE, min_sim, max_sim, ATT_INDEXES,
                                                             datapt_hash, min_cos_sim, tot_copy, simf)
            # random_tuples0 =csvTable2datasetRANDOM_bilancedWITHlsh(TABLE1_FILE, TABLE2_FILE, 2000, min_sim, max_sim, ATT_INDEXES,datapt_hash, simf )
            # tot_pt=86#len(datapt_hash)*2
            # random_tuples=csvTable2datasetRANDOM_nomatch(TABLE1_FILE, TABLE2_FILE, tot_pt, min_sim, ATT_INDEXES, simf )
            # for i in range(len(datapt_hash)):
            # if datapt_hash[i] not in random_tuples0:
            # random_tuples0.append(datapt_hash[i])
            # random_tuples0.extend(datapt_hash)
            # print("match: "+str(match)+"; no_match: "+str(no_match)+"; nlog1: "+str(nlog1)+"; nlog2: "+str(nlog2))
            random.shuffle(random_tuples0)
            random_tuples0sort = sorted(random_tuples0, key=lambda tup: (tup[2][0]))
            plot_pretrain(random_tuples0sort)

            total_ptData = len(datapt_hash)

            random_tuples1 = random_tuples0sort[:k]
            random_tuples2 = random_tuples0sort[-k:]

            random_tuples1 += random_tuples2

            # Concatenazione.
            vinsim_data += random_tuples1

            # vinsim_data += random_tuples
            # Shuffle.
            shuffle(vinsim_data)


            # Split in training set e validation set.
            def split_training_valid(data, SPLIT_FACTOR=0.8):
                bound = int(len(data) * SPLIT_FACTOR)
                train = data[:bound]
                valid = data[bound:]

                return train, valid

            plt.xlabel(get_lambda_name(simf))

            t, sim_list = plot_dataPT(vinsim_data)
            plt.xlabel('')
            #t = t[:tot_copy:-tot_copy]
            #sim_list = sim_list[:tot_copy:-tot_copy]
            gradino = []
            for g in range(len(t)):
                if g >= len(t) / 2:
                    gradino.append(1)
                else:
                    gradino.append(0)
            mse = (np.square(np.array(gradino) - sim_list)).mean(axis=None)
            if (mse < lowestMSE):
                lowestMSE = mse
                bestFun = simf
                print("update: function="+get_lambda_name(bestFun)+"', MSE="+str(lowestMSE))

    print("best function is '"+get_lambda_name(bestFun)+"' with MSE="+str(lowestMSE))
    best.append([get_lambda_name(bestFun), lowestMSE])

print(best)