# -*- coding: utf-8 -*-
"""
"""

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


def training(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf, soglia, tot_copy):
    # Imposta manualmente a True per caricare da disco tutti i modelli salvati.
    # Imposta manualmente a False per ri-eseguire tutti gli addestramenti.
    LOAD_Dataset_FROM_DISK = False
    LOAD_FROM_DISK = False
    # Il nome con cui saranno etichettati i files prodotti

    # Caricamento dati e split iniziale.
    if LOAD_Dataset_FROM_DISK:

        # Carica dataset salvato su disco.
        print("non salva i dataset")
        # data = uls.load_list('dataset_{}'.format(DATASET_NAME))
        data = []

    else:

        # Crea il dataset.
        data = csv_2_datasetALTERNATE(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)

        print(data[0])
        '''
        for i in range(len(data)):
            if data[i][3] == 1:
                print(data[i][2])
        '''
        min_sim_Match, max_sim_noMatch = plot_graph(data)
        print("min_sim_Match " + str(min_sim_Match) + "max_sim_noMatch " + str(max_sim_noMatch))
        max_sim = soglia + max(min_sim_Match, max_sim_noMatch)
        if max_sim > 0.94:
            max_sim = 0.82

        print("!max_sim " + str(max_sim))
        min_sim = min(min_sim_Match, max_sim_noMatch)  # -soglia
        print("!min_sim " + str(min_sim))
        # Salva dataset su disco.
        # uls.save_list(data, 'dataset_{}'.format(DATASET_NAME))

    # Unflat data and cut too long attributes
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
                #print('there is a list ! -> '+str(len(lb)))
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

    # Avvio addestramenti o carica da disco.
    dm_model_5, trainLab_5, validationLab_5, testLab_5 = train_dm(DATASET_NAME, train, valid, test, 5)
    dm_model_10, trainLab_10, validationLab_10, testLab_10 = train_dm(DATASET_NAME, train, valid, test, 10)
    dm_model_25, trainLab_25, validationLab_25, testLab_25 = train_dm(DATASET_NAME, train, valid, test, 25)
    dm_model_50, trainLab_50, validationLab_50, testLab_50 = train_dm(DATASET_NAME, train, valid, test, 50)
    dm_model_75, trainLab_75, validationLab_75, testLab_75 = train_dm(DATASET_NAME, train, valid, test, 75)
    dm_model_100, trainLab_100, validationLab_100, testLab_100 = train_dm(DATASET_NAME, train, valid, test, 100)
    dm_model_200, trainLab_200, validationLab_200, testLab_200 = train_dm(DATASET_NAME, train, valid, test, 200)

    f_list = []
    f_list.append(eval_dm(dm_model_200, testLab_200))
    f_list.append(eval_dm(dm_model_100, testLab_100))
    f_list.append(eval_dm(dm_model_75, testLab_75))
    f_list.append(eval_dm(dm_model_50, testLab_50))
    f_list.append(eval_dm(dm_model_25, testLab_25))
    f_list.append(eval_dm(dm_model_10, testLab_10))
    f_list.append(eval_dm(dm_model_5, testLab_5))

    def write_file_fScore(datasetname, funsim, fscore_list, model_name, metric):
        file1 = open("{a}_{c}_{b}_{d}.txt".format(a=datasetname, b=funsim, c=metric, d=model_name), "a")
        file1.write(model_name)
        file1.write("tuplecount_ft " + str(tuplecount_ft) + '\n')
        for element in fscore_list:
            file1.write(str(element))
            file1.write(' ')
        file1.write('\n')

        file1.close()

    model = 'dm'
    tuplecount_ft = [200, 100, 75, 50, 25, 10, 5]

    write_file_fScore(DATASET_NAME, funsimstr, f_list, model, "f_score")

    print("create vinsim dataset")

    # Generazione di tuple random.
    # random_tuples0,match,no_match,nlog1,nlog2=csvTable2datasetRANDOM_extremeRIP(TABLE1_FILE, TABLE2_FILE, ATT_INDEXES,ripA,ripB, simf)
    # random_tuples0,match,no_match,nlog1,nlog2 =csvTable2datasetRANDOM_extreme(TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf )
    # random_tuples0 =csvTable2datasetRANDOM_bilanced(TABLE1_FILE, TABLE2_FILE, tot_pt, min_sim, max_sim, ATT_INDEXES, simf )
    # random_tuples = csvTable2datasetRANDOM(TABLE1_FILE, TABLE2_FILE, len(data)*2, ATT_INDEXES, simf)
    datapt_hash = minHash_lsh(TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)

    def trainK(k):
        print('training with k='+str(k))
        # Dataset per VinSim.
        vinsim_data = []

        # Porzione di tuple in match da includere nell'addestramento di VinSim.
        TP_FACTOR = 0.05

        # Preleva solo quelle in match con il relativo sim vector.
        #sim_data = unflat(data, 3)
        for i in range(len(data)):
            if data[i][3] == 1:
                r = data[i]
                vinsim_data.append(r)


        # Taglio della porzione desiderata.
        # bound = int(len(vinsim_data) * TP_FACTOR)
        bound = 5
        vinsim_data = vinsim_data[:bound]

        min_cos_sim = min_cos(vinsim_data)
        print("min_cos_sim " + str(min_cos_sim))

        random_tuples0 = csvTable2datasetRANDOM_likeGold(TABLE1_FILE, TABLE2_FILE, min_sim, max_sim, ATT_INDEXES,
                                                         datapt_hash, min_cos_sim, tot_copy, simf)
        # random_tuples0 =csvTable2datasetRANDOM_bilancedWITHlsh(TABLE1_FILE, TABLE2_FILE, 2000, min_sim, max_sim, ATT_INDEXES,datapt_hash, simf )
        # tot_pt=86#len(datapt_hash)*2
        print("tot_pt: " + str(tot_pt))
        print("len(datapt_hash) " + str(len(datapt_hash)))
        print("len(random_tuples0) " + str(len(random_tuples0)))
        # random_tuples=csvTable2datasetRANDOM_nomatch(TABLE1_FILE, TABLE2_FILE, tot_pt, min_sim, ATT_INDEXES, simf )
        # for i in range(len(datapt_hash)):
        # if datapt_hash[i] not in random_tuples0:
        # random_tuples0.append(datapt_hash[i])
        # random_tuples0.extend(datapt_hash)
        # print("match: "+str(match)+"; no_match: "+str(no_match)+"; nlog1: "+str(nlog1)+"; nlog2: "+str(nlog2))
        random.shuffle(random_tuples0)
        random_tuples0sort = sorted(random_tuples0, key=lambda tup: (tup[2][0]))
        print("---------------- RANDOM TUPLES -------------------------")
        plot_pretrain(random_tuples0sort)

        total_ptData = len(datapt_hash)

        random_tuples1 = random_tuples0sort[:k]
        print("random_tuples1[:10]")
        print(random_tuples1[:10])
        print("random_tuples1[-10:]")
        print(random_tuples1[-10:])
        random_tuples2 = random_tuples0sort[-k:]
        print("random_tuples2[:10]")
        print(random_tuples2[:10])
        print("random_tuples2[-10:]")
        print(random_tuples2[-10:])

        random_tuples1 += random_tuples2

        print(len(random_tuples1))
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

        plot_dataPT(vinsim_data)

        # Filtro.
        vinsim_data = unflat(vinsim_data, 2)

        sim_train, sim_valid = split_training_valid(vinsim_data)

        all_train = join(train, valid, test, sim_train, sim_valid, DATASET_NAME)

        # Inizializza un nuovo modello.
        finetuned_model_5 = pt_ft_dm_classifier(all_train, DATASET_NAME, sim_train, sim_valid, trainLab_5, validationLab_5)
        finetuned_model_10 = pt_ft_dm_classifier(all_train, DATASET_NAME, sim_train, sim_valid, trainLab_10, validationLab_10)
        finetuned_model_25 = pt_ft_dm_classifier(all_train, DATASET_NAME, sim_train, sim_valid, trainLab_25, validationLab_25)
        finetuned_model_50 = pt_ft_dm_classifier(all_train, DATASET_NAME, sim_train, sim_valid, trainLab_50, validationLab_50)
        finetuned_model_75 = pt_ft_dm_classifier(all_train, DATASET_NAME, sim_train, sim_valid, trainLab_75, validationLab_75)
        finetuned_model_100 = pt_ft_dm_classifier(all_train, DATASET_NAME, sim_train, sim_valid, trainLab_100, validationLab_100)
        finetuned_model_200 = pt_ft_dm_classifier(all_train, DATASET_NAME, sim_train, sim_valid, trainLab_200, validationLab_200)

        f_list = []
        f_list.append(eval_dm(finetuned_model_200, testLab_200))
        f_list.append(eval_dm(finetuned_model_100, testLab_100))
        f_list.append(eval_dm(finetuned_model_75, testLab_75))
        f_list.append(eval_dm(finetuned_model_50, testLab_50))
        f_list.append(eval_dm(finetuned_model_25, testLab_25))
        f_list.append(eval_dm(finetuned_model_10, testLab_10))
        f_list.append(eval_dm(finetuned_model_5, testLab_5))

        def write_file_fScore(datasetname, funsim, fscore_list, model_name, metric, total_tup_pt):
            file1 = open("{a}_{c}_{b}_{d}.txt".format(a=datasetname, b=funsim, c=metric, d=model_vin), "a")
            file1.write(model_name)
            file1.write("total_tup_preTraining " + str(total_tup_pt) + '\n')
            for element in fscore_list:
                file1.write(str(element))
                file1.write(' ')
            file1.write('\n')

            file1.close()

        model_vin = 'vinsim'

        total_tup_pt = len(vinsim_data)

        tuplecount_ft = [200, 100, 75, 50, 25, 10, 5]

        write_file_fScore(DATASET_NAME, funsimstr, f_list, model_vin, "f_score", total_tup_pt)

    kappa = [int(tot_pt / 2)]  # 100,250,500,1000]
    # kappa=[tot_pt]#100,250,500,1000]
    for i in range(len(kappa)):
        trainK(kappa[i])


DATASET_NAME = 'fodo_zaga'
GROUND_TRUTH_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/matches_fodors_zagats.csv'
TABLE1_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/fodors.csv'
TABLE2_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/zagats.csv'
ATT_INDEXES = [(1, 1), (2, 2), (3, 3),(4,4), (5, 5), (6, 6)]

simf = lambda a, b: sim_function.sim_bf_fz2b(a, b)
funsimstr = "sim_bf_fz2b"

tot_pt = 2000  # dimensione dataset pre_training
tot_copy = 900 # numero di elementi generati con edit distance
soglia = 0.03  # da aggiungere per discostarsi da min_sim e max_sim ottenuto
runs = 3
for i in range(runs):
    training(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf, soglia, tot_copy)
