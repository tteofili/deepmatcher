# -*- coding: utf-8 -*-
"""
"""
from __future__ import print_function

import deepmatcher as dm
import pandas as pd
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
from dm_train import pt_ft_dm_classifier, pt_ft_dm_full
from dm_train import join
import sim_function
from brute_force_aggregate_sim_find import bf, bf2
from sim_function_find import find_sim
from inspect import getsource
import random
import string

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

simfunctions = [
    lambda t1, t2: sim_function.jaro(t1, t2),
    lambda t1, t2: sim_function.sim_jacc(t1, t2),
    lambda t1, t2: sim_function.sim_cos(t1, t2),
    lambda t1, t2: sim_function.sim_hamming(t1, t2),
    lambda t1, t2: sim_function.sim_lcs(t1, t2),
    lambda t1, t2: sim_function.sim_lev(t1, t2),
    lambda t1, t2: sim_function.sim_ngram(t1, t2),
    lambda t1, t2: sim_function.sim_sodi(t1, t2),
    lambda t1, t2: sim_function.sim_sodi(t1.split(), t2.split()),
    lambda t1, t2: sim_function.jaro(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_cos(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_hamming(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_lcs(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_lev(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_ngram(t1.split(), t2.split()),
]
'''
lambda t1, t2: sim_function.sim_bert(t1, t2),
    lambda t1, t2: sim_function.sim_sbert(t1, t2),
    lambda t1, t2: sim_function.sim_sbert2(t1, t2),
    '''

get_lambda_name = lambda l: getsource(l).split('=')[0].strip()


def write_file_fScore(datasetname, funsim, fscore_list, model_name, metric, total_tup_pt):
    file1 = open("{a}_{c}_{b}_{d}.txt".format(a=datasetname, b=funsim, c=metric, d=model_name), "a")
    file1.write(model_name)
    file1.write("total_tup_preTraining " + str(total_tup_pt) + '\n')
    for element in fscore_list:
        file1.write(str(element))
        file1.write(' ')
    file1.write('\n')

    file1.close()

def cut(name, train, valid, test, cut, datadir):
    if (len(train) > cut):
        train = train[:cut]

    names = []
    names.append('id')
    names.append('label')
    attr_per_tab = int((len(train[0]) - 2) / 2)
    for i in range(attr_per_tab):
        names.append('left_attr_' + str(i))
    for i in range(attr_per_tab):
        names.append('right_attr_' + str(i))

    df = pd.DataFrame(train)
    df.columns = names
    df.to_csv(datadir + name + '/train_' + str(cut) + '.csv', index=False)

    df = pd.DataFrame(test)
    df.columns = names
    df.to_csv(datadir + name + '/test_' + str(cut) + '.csv', index=False)

    df = pd.DataFrame(valid)
    df.columns = names
    df.to_csv(datadir + name + '/valid_' + str(cut) + '.csv', index=False)

    # read dataset
    trainLab, validationLab, testLab = dm.data.process(path=datadir + name,
                                                       train='train_' + str(cut) + '.csv',
                                                       validation='valid_' + str(cut) + '.csv',
                                                       test='test_' + str(cut) + '.csv')
    return trainLab, validationLab, testLab


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

# Split in train, validation and test set.
def split_training_valid_test(data, SPLIT_FACTOR=0.8):
    bound = int(len(data) * SPLIT_FACTOR)
    train = data[:bound]
    remain = data[bound:]
    bound2 = int(len(remain) * 0.5)
    test = remain[bound2:]
    valid = remain[:bound2]

    return train, valid, test

'''
def write_file_fScore(datasetname, funsim, fscore_list, model_name, metric):
    tuplecount_ft = [200, 100, 75, 50, 25, 10, 5]
    file1 = open("{a}_{c}_{b}_{d}.txt".format(a=datasetname, b=funsim, c=metric, d=model_name), "a")
    file1.write(model_name)
    file1.write("tuplecount_ft " + str(tuplecount_ft) + '\n')
    for element in fscore_list:
        file1.write(str(element))
        file1.write(' ')
    file1.write('\n')

    file1.close()'''


# Split in training set e validation set.
def split_training_valid(data, SPLIT_FACTOR=0.8):
    bound = int(len(data) * SPLIT_FACTOR)
    train = data[:bound]
    valid = data[bound:]

    return train, valid

def trainK(k, data, min_sim, max_sim, t1_file, t2_file, indexes, dataset_name, datadir, simf):
    print('training with k=' + str(k))

    deepmatcher_data = unflat(data, 3)

    # Tutti i successivi addestramenti partiranno dal 100% di train (80% di tutti i dati).
    # Le tuple in valid non verranno mai usate per addestrare ma solo per testare i modelli.
    train, valid, test = split_training_valid_test(deepmatcher_data)

    trainLab_5, validationLab_5, testLab_5 = cut(dataset_name, train, valid, test, 5, datadir)
    trainLab_10, validationLab_10, testLab_10 = cut(dataset_name, train, valid, test, 10, datadir)
    trainLab_25, validationLab_25, testLab_25 = cut(dataset_name, train, valid, test, 25, datadir)
    trainLab_50, validationLab_50, testLab_50 = cut(dataset_name, train, valid, test, 50, datadir)
    trainLab_75, validationLab_75, testLab_75 = cut(dataset_name, train, valid, test, 75, datadir)
    trainLab_100, validationLab_100, testLab_100 = cut(dataset_name, train, valid, test, 100, datadir)
    trainLab_200, validationLab_200, testLab_200 = cut(dataset_name, train, valid, test, 200, datadir)

    print("create vinsim dataset")

    # Generazione di tuple random.
    # random_tuples0,match,no_match,nlog1,nlog2=csvTable2datasetRANDOM_extremeRIP(TABLE1_FILE, TABLE2_FILE, ATT_INDEXES,ripA,ripB, simf)
    # random_tuples0,match,no_match,nlog1,nlog2 =csvTable2datasetRANDOM_extreme(TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf )
    # random_tuples0 =csvTable2datasetRANDOM_bilanced(TABLE1_FILE, TABLE2_FILE, tot_pt, min_sim, max_sim, ATT_INDEXES, simf )
    # random_tuples = csvTable2datasetRANDOM(TABLE1_FILE, TABLE2_FILE, len(data)*2, ATT_INDEXES, simf)
    datapt_hash = minHash_lsh(t1_file, t2_file, indexes, simf)


    # Dataset per VinSim.
    vinsim_data = []

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
    print("min_cos_sim " + str(min_cos_sim))

    random_tuples0 = csvTable2datasetRANDOM_likeGold(t1_file, t2_file, min_sim, max_sim, indexes,
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

    plot_dataPT(vinsim_data)

    # Filtro.
    vinsim_data = unflat(vinsim_data, 2)

    sim_train, sim_valid = split_training_valid(vinsim_data)

    all_train = join(train, valid, test, sim_train, sim_valid, dataset_name)

    # Inizializza un nuovo modello.
    finetuned_model_5 = pt_ft_dm_full(all_train, dataset_name, sim_train, sim_valid, trainLab_5, validationLab_5)
    finetuned_model_10 = pt_ft_dm_full(all_train, dataset_name, sim_train, sim_valid, trainLab_10,
                                             validationLab_10)
    finetuned_model_25 = pt_ft_dm_full(all_train, dataset_name, sim_train, sim_valid, trainLab_25,
                                             validationLab_25)
    finetuned_model_50 = pt_ft_dm_full(all_train, dataset_name, sim_train, sim_valid, trainLab_50,
                                             validationLab_50)
    finetuned_model_75 = pt_ft_dm_full(all_train, dataset_name, sim_train, sim_valid, trainLab_75,
                                             validationLab_75)
    finetuned_model_100 = pt_ft_dm_full(all_train, dataset_name, sim_train, sim_valid, trainLab_100,
                                              validationLab_100)
    finetuned_model_200 = pt_ft_dm_full(all_train, dataset_name, sim_train, sim_valid, trainLab_200,
                                              validationLab_200)

    f_list = []
    f_list.append(eval_dm(finetuned_model_200, testLab_200))
    f_list.append(eval_dm(finetuned_model_100, testLab_100))
    f_list.append(eval_dm(finetuned_model_75, testLab_75))
    f_list.append(eval_dm(finetuned_model_50, testLab_50))
    f_list.append(eval_dm(finetuned_model_25, testLab_25))
    f_list.append(eval_dm(finetuned_model_10, testLab_10))
    f_list.append(eval_dm(finetuned_model_5, testLab_5))

    model_vin = 'vinsim'

    total_tup_pt = len(vinsim_data)

    print('WRITING RESULTS')
    write_file_fScore(dataset_name, simf, f_list, model_vin, "f_score", total_tup_pt)

def pretrain(gt_file, t1_file, t2_file, indexes, simf, soglia, tot_copy, dataset_name, datadir):
    # Imposta manualmente a True per caricare da disco tutti i modelli salvati.
    # Imposta manualmente a False per ri-eseguire tutti gli addestramenti.
    LOAD_Dataset_FROM_DISK = False
    LOAD_FROM_DISK = False
    # Il nome con cui saranno etichettati i files prodotti

    # Crea il dataset.
    data = csv_2_datasetALTERNATE(gt_file, t1_file, t2_file, indexes, simf)

    print(data[0])

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

    kappa = [int(tot_pt / 2)]
    for i in range(len(kappa)):
        trainK(kappa[i], data, min_sim, max_sim, t1_file, t2_file, indexes, dataset_name, datadir, simf)


base_dir = '/home/tteofili/Downloads/dataset/'
'''
DATASET_NAME = 'fodo_zaga'
GROUND_TRUTH_FILE = '%s' + DATASET_NAME + '/matches_fodors_zagats.csv'
TABLE1_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/fodors.csv'
TABLE2_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/zagats.csv'
ATT_INDEXES = [(1, 1), (2, 2), (3, 3),(4,4), (5, 5), (6, 6)]
''' % base_dir

'''DATASET_NAME = 'dirty_dblp_scholar'
GROUND_TRUTH_FILE = '%s' + DATASET_NAME + '/all.csv'
TABLE1_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/tableA.csv'
TABLE2_FILE = '/home/tteofili/Downloads/dataset/' + DATASET_NAME + '/tableB.csv'
ATT_INDEXES = [(1, 1), (2, 2), (3, 3), (4, 4)]
''' % base_dir


tot_pt = 2000  # dimensione dataset pre_training
tot_copy = 900 # numero di elementi generati con edit distance
soglia = 0.01  # da aggiungere per discostarsi da min_sim e max_sim ottenuto
runs = 1

datasets = [
    [('%sabt_buy_anhai/all.csv' % base_dir), ('%sabt_buy_anhai/tableA.csv' % base_dir),
     ('%sabt_buy_anhai/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3)], 'abt_buy_anhai',
     ('%scustom/' % base_dir)],
    [('%sfodo_zaga/matches_fodors_zagats.csv' % base_dir),
     ('%sfodo_zaga/fodors.csv' % base_dir),
     ('%sfodo_zaga/zagats.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], 'fodo_zaga',
     ('%scustom/' % base_dir)],
    [('%sdirty_dblp_scholar/all.csv' % base_dir), ('%sdirty_dblp_scholar/tableA.csv' % base_dir),
     ('%sdirty_dblp_scholar/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dirty_dblp_scholar',
     ('%scustom/' % base_dir)],
    [('%sdirty_dblp_acm/all.csv' % base_dir), ('%sdirty_dblp_acm/tableA.csv' % base_dir),
     ('%sdirty_dblp_acm/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dirty_dblp_acm',
     ('%scustom/' % base_dir)],
    [('%sdirty_walmart_amazon/all.csv' % base_dir), ('%sdirty_walmart_amazon/tableA.csv' % base_dir),
     ('%sdirty_walmart_amazon/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dirty_walmart_amazon',
     ('%scustom/' % base_dir)],
    [('%sdirty_amazon_itunes/all.csv' % base_dir), ('%sdirty_amazon_itunes/tableA.csv' % base_dir),
     ('%sdirty_amazon_itunes/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dirty_amazon_itunes',
     ('%scustom/' % base_dir)],
    [('%sdblp_scholar/DBLP-Scholar-perfectMapping.csv' % base_dir), ('%sdblp_scholar/DBLP1.csv' % base_dir),
     ('%sdblp_scholar/Scholar.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dblp_scholar',
     ('%scustom/' % base_dir)],
    [('%samazon_google/Amazon_GoogleProducts-perfectMapping.csv' % base_dir),
     ('%samazon_google/AmazonAG.csv' % base_dir),
     ('%sdblp_scholar/amazon_google.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'amazon_google',
     ('%scustom/' % base_dir)],
    [('%sbeers/all.csv' % base_dir), ('%sbeers/tableA.csv' % base_dir),
     ('%sbeers/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'beers',
     ('%scustom/' % base_dir)],
    [('%swalmart_amazon/matches_walmart_amazon.csv' % base_dir), ('%swalmart_amazon/walmart.csv' % base_dir),
     ('%swalmart_amazon/amazonw.csv' % base_dir), [(5, 9), (4, 5),(3, 3),(14, 4),(6, 11)], 'walmart_amazon',
     ('%scustom/' % base_dir)],
]

def generate_samples(indexes):
    s1 = []
    s2 = []
    for ii in indexes:
        s1.append(randomString(8))
        s2.append(randomString(8))
    return s1, s2

def create_lc_sim(best_sims, indexes):
    print('generating function')
    rs = []
    for i in range(len(indexes)):
        s1, s2 = generate_samples(indexes)
        left_index = indexes[i][0] -1
        right_index = indexes[i][1] -1
        lambdas = []
        for j in range(len(best_sims[i])):
            localsim = best_sims[i][j][0]
            localweight = best_sims[i][j][1]
            #print(f'on {left_index},{right_index}: {localsim} w:{localweight}')
            sim_lambda = lambda t1, t2: localsim(t1[left_index], t2[right_index])[0] * localweight
            lambdas.append(sim_lambda)
            try:
                print(f'trying {sim_lambda(s1,s2)}')
            except:
                pass
        aggr = lambda t1, t2: np.sum(np.array([sim(t1, t2) for sim in lambdas]))
        try:
            print(f'trying {aggr(s1,s2)}')
        except:
            pass
        rs.append(aggr)
    finalf = lambda t1, t2: [np.sum(np.array([r(t1, t2) for r in rs]))/len(rs)]
    return finalf


def create_single_sim(bf_fun):
    per_att_sim = lambda t1, t2: [np.sum(np.array([bf_fun[i](t1[i], t2[i]) for i in range(len(bf_fun))]))/len(bf_fun)]
    return per_att_sim


for i in range(runs):
    for d in datasets:
        gt_file = d[0]
        t1_file = d[1]
        t2_file = d[2]
        indexes = d[3]
        dataset_name = d[4]
        datadir = d[5]
        print(f'---{dataset_name}---')
        allsims = simfunctions.copy()
        print('finding best linear combination per attribute function')
        best_sims = bf2(gt_file, t1_file, t2_file, indexes, allsims)
        fsims = []
        ind = 0
        for bsk in best_sims:
            top_sims_k = bsk[:3]
            sw = np.sum(np.array(top_sims_k)[:,1])
            for w in top_sims_k:
                w[1] = w[1]/sw
            print(f'for attributes {indexes[ind]}:')
            for bsa in top_sims_k:
                print(f'{get_lambda_name(bsa[0])}, w:{bsa[1]}')
            fsims.append(top_sims_k)
            ind += 1
        generated_sim = create_lc_sim(fsims, indexes)
        print('finding best single per attribute function')
        bf_fun = bf(gt_file, t1_file, t2_file, indexes, allsims)
        generated_sim_single = create_single_sim(bf_fun)
        allsims = [generated_sim] + [generated_sim_single]
        print('looking for best function')
        sf = find_sim(gt_file, t1_file, t2_file, indexes, allsims, 0, 100, 30)
        for s in sf:
            print(f'pt with {get_lambda_name(s)}')
            pretrain(gt_file, t1_file, t2_file, indexes, s, soglia, tot_copy, dataset_name, datadir)