# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:00:08 2019

@author: Giulia
"""
import textdistance
import textdistance as txd
from strsimpy.metric_lcs import MetricLCS
from strsimpy.ngram import NGram
import nltk
import re, math
from collections import Counter
import torch
import scipy
import numpy


WORD = re.compile(r'\w+')


def concatenate_list_data(list):
    result = ''
    for element in list:
        result += ' ' + str(element)
    return result


# calcola la cos similarity di due vettori
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


'''EDIT SIM'''


def sim_lev(tuple1, tuple2):
    t1_concat = concatenate_list_data(tuple1)
    t2_concat = concatenate_list_data(tuple2)

    lev = textdistance.levenshtein.normalized_similarity(t1_concat, t2_concat)
    vector = [lev]
    return vector


def jaro(tuple1, tuple2):
    t1_concat = concatenate_list_data(tuple1)
    t2_concat = concatenate_list_data(tuple2)
    jawi = txd.jaro_winkler.normalized_similarity(t1_concat, t2_concat)
    vector = [jawi]
    return vector


def sim_ngram(tuple1, tuple2):
    t1_concat = concatenate_list_data(tuple1)
    t2_concat = concatenate_list_data(tuple2)
    ngram = NGram()
    ngram1 = 1 - ngram.distance(t1_concat, t2_concat)
    vector = [ngram1]
    return vector


def sim_lcs(tuple1, tuple2):
    t1_concat = concatenate_list_data(tuple1)
    t2_concat = concatenate_list_data(tuple2)
    metric_lcs = MetricLCS()
    lcs1 = 1 - metric_lcs.distance(t1_concat, t2_concat)
    vector = [lcs1]
    return vector


def sim_hamming(tuple1, tuple2):
    t1_concat = concatenate_list_data(tuple1)
    t2_concat = concatenate_list_data(tuple2)
    ham = txd.hamming.normalized_similarity(t1_concat, t2_concat)
    vector = [ham]
    return vector


def jacc_trigram(tupla1, tupla2):
    sent1 = concatenate_list_data(tupla1)
    sent2 = concatenate_list_data(tupla2)

    ng1_chars = set(nltk.ngrams(sent1, n=3))
    ng2_chars = set(nltk.ngrams(sent2, n=3))

    jd_sent_1_2 = 1 - nltk.jaccard_distance(ng1_chars, ng2_chars)
    vector = [jd_sent_1_2]
    return vector


'''TOKEN SIM'''

import nltk


def sim_cos(tuple1, tuple2):
    stringa1 = concatenate_list_data(tuple1)
    stringa2 = concatenate_list_data(tuple2)
    cos_sim = get_cosine(text_to_vector(stringa1), text_to_vector(stringa2))
    vector = [cos_sim]
    return vector


def sim_jacc(tuple1, tuple2):
    t1_concat = concatenate_list_data(tuple1)
    t2_concat = concatenate_list_data(tuple2)
    t1_split = t1_concat.split()
    t2_split = t2_concat.split()
    jacc = textdistance.jaccard.normalized_similarity(t1_split, t2_split)
    vector = [jacc]
    return vector

def sim_equals(tuple1, tuple2):
    return [tuple1 == tuple2]

def sim_noselect(tuple1, tuple2):
    return [0.5]

def sim_sodi(tuple1, tuple2):
    t1_concat = concatenate_list_data(tuple1)
    t2_concat = concatenate_list_data(tuple2)
    t1_split = t1_concat.split()
    t2_split = t2_concat.split()
    sodi = textdistance.sorensen_dice.normalized_similarity(t1_split, t2_split)
    vector = [sodi]
    return vector


def jacc_trigramTOKEN(tupla1, tupla2):
    sent1 = concatenate_list_data(tupla1)
    sent2 = concatenate_list_data(tupla2)

    tokens1 = nltk.word_tokenize(sent1)
    tokens2 = nltk.word_tokenize(sent2)

    ng1_tokens = set(nltk.ngrams(tokens1, n=3))
    ng2_tokens = set(nltk.ngrams(tokens2, n=3))

    jd_sent_1_2 = 1 - nltk.jaccard_distance(ng1_tokens, ng2_tokens)
    vector = [jd_sent_1_2]
    return vector


def remove_symb(tupla):
    tupla1 = []
    for el in tupla:
        t = re.sub(r'[^\w]', ' ', str(el))
        tupla1.append(t)
    print(tupla1)

    return tupla1


'''sim4attr'''


def sim4attrFZ(stringa1, stringa2):
    s0 = txd.jaro_winkler.normalized_similarity(stringa1[0], stringa2[0])

    t1_split = stringa1[1].split()
    t2_split = stringa2[1].split()
    s1 = textdistance.jaccard.normalized_similarity(t1_split, t2_split)

    s2 = get_cosine(text_to_vector(stringa1[2]), text_to_vector(stringa2[2]))
    s3 = textdistance.levenshtein.normalized_similarity(stringa1[3], stringa2[3])
    t1_split4 = stringa1[4].split()
    t2_split4 = stringa2[4].split()
    s4 = textdistance.jaccard.normalized_similarity(t1_split4, t2_split4)

    s5 = textdistance.levenshtein.normalized_similarity(stringa1[5], stringa2[5])
    vect = [s0, s1, s2, s3, s4, s5]
    # print(vect)
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    aver = round(sum(vect) / len(vect), 2)

    return [aver]


def sim4attrFZwoClassEPhone(stringa1, stringa2):
    s0 = txd.jaro_winkler.normalized_similarity(stringa1[0], stringa2[0])

    t1_split = stringa1[1].split()
    t2_split = stringa2[1].split()
    s1 = textdistance.jaccard.normalized_similarity(t1_split, t2_split)

    s2 = get_cosine(text_to_vector(stringa1[2]), text_to_vector(stringa2[2]))
    # s3=textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    t1_split4 = stringa1[3].split()
    t2_split4 = stringa2[3].split()
    s3 = textdistance.jaccard.normalized_similarity(t1_split4, t2_split4)

    # s5=textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    vect = [s0, s1, s2, s3]  # ,s4]#,s5]
    # print(vect)
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    aver = round(sum(vect) / len(vect), 2)

    return [aver]


def sim4attrFZwoClass(stringa1, stringa2):
    s0 = txd.jaro_winkler.normalized_similarity(stringa1[0], stringa2[0])

    t1_split = stringa1[1].split()
    t2_split = stringa2[1].split()
    s1 = textdistance.jaccard.normalized_similarity(t1_split, t2_split)

    s2 = get_cosine(text_to_vector(stringa1[2]), text_to_vector(stringa2[2]))
    s3 = textdistance.levenshtein.normalized_similarity(stringa1[3], stringa2[3])
    t1_split4 = stringa1[4].split()
    t2_split4 = stringa2[4].split()
    s4 = textdistance.jaccard.normalized_similarity(t1_split4, t2_split4)

    # s5=textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    vect = [s0, s1, s2, s3, s4]  # ,s5]
    # print(vect)
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    aver = round(sum(vect) / len(vect), 2)

    return [aver]


def sim4attrWA(stringa1, stringa2):
    # t10_split=stringa1[0].split()
    # t20_split=stringa2[0].split()
    # s0 = textdistance.sorensen_dice.normalized_similarity(t10_split,t20_split)
    # s0=txd.jaro_winkler.normalized_similarity(stringa1[0],stringa2[0])
    s0 = get_cosine(text_to_vector(stringa1[0]), text_to_vector(stringa2[0]))

    # t1_split=stringa1[1].split()
    # t2_split=stringa2[1].split()
    # s1= textdistance.sorensen_dice.normalized_similarity(t1_split,t2_split)
    s1 = get_cosine(text_to_vector(stringa1[1]), text_to_vector(stringa2[1]))

    s2 = textdistance.levenshtein.normalized_similarity(stringa1[2], stringa2[2])
    s3 = get_cosine(text_to_vector(stringa1[3]), text_to_vector(stringa2[3]))
    # s3= textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    s4 = textdistance.levenshtein.normalized_similarity(stringa1[4], stringa2[4])
    vect = [s0, s1, s2, s3, s4]
    # print(vect)
    rm_min = min(vect)
    vect.remove(rm_min)
    # print(vect)

    aver = round(sum(vect) / len(vect), 2)
    # print(aver)
    return [aver]


def sim4attrGA(stringa1, stringa2):
    t10_split = stringa1[0].split()
    t20_split = stringa2[0].split()
    s0 = textdistance.sorensen_dice.normalized_similarity(t10_split, t20_split)
    # s0=txd.jaro_winkler.normalized_similarity(stringa1[0],stringa2[0])

    t1_split = stringa1[1].split()
    t2_split = stringa2[1].split()
    s1 = textdistance.sorensen_dice.normalized_similarity(t1_split, t2_split)

    s2 = textdistance.levenshtein.normalized_similarity(stringa1[2], stringa2[2])
    s3 = textdistance.levenshtein.normalized_similarity(stringa1[3], stringa2[3])

    vect = [s0, s1, s2, s3]
    # print(vect)
    rm_min = min(vect)
    vect.remove(rm_min)
    # print(vect)

    aver = round(sum(vect) / len(vect), 2)
    # print(aver)
    return [aver]


def sim4attrScho(stringa1, stringa2):
    s0 = txd.jaro_winkler.normalized_similarity(stringa1[0], stringa2[0])

    t1_split = stringa1[1].split()
    t2_split = stringa2[1].split()
    s1 = textdistance.jaccard.normalized_similarity(t1_split, t2_split)

    s2 = get_cosine(text_to_vector(stringa1[2]), text_to_vector(stringa2[2]))
    s3 = textdistance.levenshtein.normalized_similarity(stringa1[3], stringa2[3])

    vect = [s0, s1, s2, s3]
    # print(vect)
    rm_min = min(vect)
    vect.remove(rm_min)
    # print(vect)
    # aver=sum(vect) / len(vect)
    aver = round(sum(vect) / len(vect), 2)
    # print(aver)
    return [aver]


def sim_bf_scho(stringa1, stringa2):
    s0 = sim_ngram(stringa1[0].split(), stringa2[0].split())

    t1_split = stringa1[1].split()
    t2_split = stringa2[1].split()
    s1 = sim_lev(t1_split, t2_split)

    s2 = sim_ngram(stringa1[2].split(), stringa2[2].split())
    s3 = sim_ngram(stringa1[3].split(), stringa2[3].split())

    vect = [s0[0], s1[0], s2[0], s3[0]]

    aver = round(sum(vect) / len(vect), 2)
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    # print(aver)
    return [aver]

def sim_bf_dirty_scho(stringa1, stringa2):
    s0 = sim_lev(stringa1[0].split(), stringa2[0].split())

    t1_split = stringa1[1].split()
    t2_split = stringa2[1].split()
    s1 = sim_sodi(t1_split, t2_split)

    s2 = jaro(stringa1[2], stringa2[2])
    s3 = jaro(stringa1[3], stringa2[3])

    vect = [s0[0], s1[0], s2[0], s3[0]]

    aver = round(sum(vect) / len(vect), 2)
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    # print(aver)
    return [aver]

def sim_bf_dirty_acm(stringa1, stringa2):
    s0 = sim_ngram(stringa1[0], stringa2[0])

    s1 = sim_jacc(stringa1[1], stringa2[1])

    s2 = jaro(stringa1[2].split(), stringa2[2].split())
    s3 = jaro(stringa1[3], stringa2[3])

    vect = [s0[0], s1[0], s2[0], s3[0]]

    aver = round(sum(vect) / len(vect), 2)
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    # print(aver)
    return [aver]


'''

'''
def sim_bf2_dda(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_cos(t1, t2)[0] * 0.54 + sim_cos(t1s, t2s)[0] * 0.46

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_hamming(t1s, t2s)[0] * 0.61 + sim_jacc(t1, t2)[0] * 0.39

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = sim_hamming(t1s, t2s)[0] * 0.53 + sim_cos(t1s, t2s)[0] * 0.47

    t1 = stringa1[3]
    t2 = stringa2[3]
    t1s = t1.split()
    t2s = t2.split()
    s3 = sim_lev(t1, t2)[0] * 0.55 + sim_ngram(t1s, t2s)[0] * 0.45

    vect = [s0, s1, s2, s3]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

'''
for attributes (1, 1):
lambda t1, t2: sim_function.sim_cos(t1, t2),, w:0.5341633925104502
lambda t1, t2: sim_function.sim_cos(t1.split(), t2.split()),, w:0.46583660748954975
for attributes (2, 2):
lambda t1, t2: sim_function.sim_hamming(t1.split(), t2.split()),, w:0.6093156983603861
lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),, w:0.39068430163961393
for attributes (3, 3):
lambda t1, t2: sim_function.sim_hamming(t1.split(), t2.split()),, w:0.5303183567307664
lambda t1, t2: sim_function.sim_cos(t1.split(), t2.split()),, w:0.4696816432692335
for attributes (5, 5):
lambda t1, t2: sim_function.sim_lev(t1, t2),, w:0.5522690126862783
lambda t1, t2: sim_function.sim_ngram(t1.split(), t2.split()),, w:0.44773098731372174
'''
def sim_bf2_fz(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_cos(t1, t2)[0] * 0.54 + sim_cos(t1s, t2s)[0] * 0.46

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_hamming(t1s, t2s)[0] * 0.61 + sim_jacc(t1, t2)[0] * 0.39

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = sim_hamming(t1s, t2s)[0] * 0.53 + sim_cos(t1s, t2s)[0] * 0.47

    t1 = stringa1[3]
    t2 = stringa2[3]
    t1s = t1.split()
    t2s = t2.split()
    s3 = sim_lev(t1, t2)[0] * 0.55 + sim_ngram(t1s, t2s)[0] * 0.45

    vect = [s0, s1, s2, s3]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

'''
for attributes (1, 1):
lambda t1, t2: sim_function.sim_cos(t1.split(), t2.split()),, w:0.5256530946456764
lambda t1, t2: sim_function.sim_sodi(t1.split(), t2.split()),, w:0.47434690535432356
for attributes (2, 2):
lambda t1, t2: sim_function.sim_sodi(t1.split(), t2.split()),, w:0.5855796779144531
lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),, w:0.4144203220855469
for attributes (3, 3):
lambda t1, t2: sim_function.sim_sodi(t1.split(), t2.split()),, w:0.5
lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),, w:0.5
'''
def sim_bf2_ab(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_cos(t1s, t2s)[0] * 0.53 + sim_sodi(t1s, t2s)[0] * 0.47

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_jacc(t1s, t2s)[0] * 0.41 + sim_sodi(t1s, t2s)[0] * 0.59

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = sim_jacc(t1s, t2s)[0] * 0.5 + sim_sodi(t1s, t2s)[0] * 0.5

    vect = [s0, s1, s2]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

'''
for attributes (5, 9):
lambda t1, t2: sim_function.sim_ngram(t1.split(), t2.split()),, w:0.5066857340495199
lambda t1, t2: sim_function.sim_cos(t1.split(), t2.split()),, w:0.4933142659504801
for attributes (4, 5):
lambda t1, t2: sim_function.sim_hamming(t1.split(), t2.split()),, w:0.6126228546335649
lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),, w:0.387377145366435
for attributes (3, 3):
lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),, w:0.5377744328904933
lambda t1, t2: sim_function.sim_hamming(t1.split(), t2.split()),, w:0.46222556710950674
for attributes (14, 4):
lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),, w:0.5418688380188083
lambda t1, t2: sim_function.sim_cos(t1, t2),, w:0.45813116198119175
for attributes (6, 11):
lambda t1, t2: sim_function.sim_hamming(t1.split(), t2.split()),, w:0.5367990893535891
lambda t1, t2: sim_function.sim_sodi(t1.split(), t2.split()),, w:0.46320091064641084
'''
def sim_bf2_wa(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_ngram(t1s, t2s)[0] * 0.51 + sim_cos(t1s, t2s)[0] * 0.49

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_hamming(t1s, t2s)[0] * 0.61 + sim_jacc(t1s, t2s)[0] * 0.39

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = sim_jacc(t1s, t2s)[0] * 0.54 + sim_hamming(t1s, t2s)[0] * 0.46

    t1 = stringa1[3]
    t2 = stringa2[3]
    t1s = t1.split()
    t2s = t2.split()
    s3 = sim_jacc(t1s, t2s)[0] * 0.54 + sim_cos(t1, t2)[0] * 0.46

    t1 = stringa1[4]
    t2 = stringa2[4]
    t1s = t1.split()
    t2s = t2.split()
    s4 = sim_hamming(t1s, t2s)[0] * 0.54 + sim_sodi(t1s, t2s)[0] * 0.46

    vect = [s0, s1, s2, s3, s4]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

'''
    for attributes (1, 1):
    lambda t1, t2: sim_function.sim_cos(t1, t2),, w:0.5357183134944616
    lambda t1, t2: sim_function.sim_cos(t1.split(), t2.split()),, w:0.46428168650553847
    for attributes (2, 2):
    lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),, w:0.5985453914917036
    lambda t1, t2: sim_function.sim_lev(t1, t2),, w:0.4014546085082964
    for attributes (3, 3):
    lambda t1, t2: sim_function.jaro(t1, t2),, w:0.5496137433499761
    lambda t1, t2: sim_function.sim_cos(t1, t2),, w:0.4503862566500239
    for attributes (4, 4):
    lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),, w:0.5938202054812824
    lambda t1, t2: sim_function.sim_sodi(t1.split(), t2.split()),, w:0.40617979451871755

'''
def sim_bf2_beers(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_cos(t1, t2)[0] * 0.54 + sim_cos(t1s, t2s)[0] * 0.46

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_jacc(t1s, t2s)[0] * 0.6 + sim_lev(t1, t2)[0] * 0.4

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = jaro(t1, t2)[0] * 0.55 + sim_cos(t1, t2)[0] * 0.45

    t1 = stringa1[3]
    t2 = stringa2[3]
    t1s = t1.split()
    t2s = t2.split()
    s3 = sim_jacc(t1s, t2s)[0] * 0.6 + sim_sodi(t1s, t2s)[0] * 0.4

    vect = [s0, s1, s2, s3]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_beers(stringa1, stringa2):

    s0 = sim_cos(stringa1[0], stringa2[0])

    s1 = sim_ngram(stringa1[1], stringa2[1])

    s2 = sim_sodi(stringa1[2], stringa2[2])
    s3 = sim_sodi(stringa1[3], stringa2[3])

    vect = [s0[0], s1[0], s2[0], s3[0]]

    aver = round(sum(vect) / len(vect), 2)
    # print(aver)
    return [aver]

def sim_bf_fz(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = sim_sodi(stringa1[0], stringa2[0]) # or ngram or cos

    s1 = sim_hamming(stringa1[1], stringa2[1]) # or cos or lev

    s2 = sim_cos(stringa1[2], stringa2[2]) # or sbert2
    s3 = sim_cos(stringa1[3], stringa2[3]) # or cos or lcs
    s4 = sim_lev(stringa1[4], stringa2[4]) # or bert or sbert2
    s5 = sim_jacc(stringa1[5], stringa2[5]) # or

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]

    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_fz2(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = sim_sodi(stringa1[0], stringa2[0])

    s1 = sim_jacc(stringa1[1], stringa2[1])

    s2 = sim_jacc(stringa1[2], stringa2[2])
    s3 = sim_ngram(stringa1[3], stringa2[3])
    s4 = jaro(stringa1[4], stringa2[4])
    s5 = sim_lev(stringa1[5], stringa2[5])

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_fz2a(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = (sim_hamming(stringa1[0].split(), stringa2[0].split()))

    s1 = sim_jacc(stringa1[1].split(), stringa2[1].split())

    s2 = sim_hamming(stringa1[2].split(), stringa2[2].split())
    s3 = sim_cos(stringa1[3].split(), stringa2[3].split())
    s4 = jaro(stringa1[4].split(), stringa2[4].split())
    s5 = sim_lev(stringa1[5].split(), stringa2[5].split())

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_fz2b(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = sim_cos(stringa1[0].split(), stringa2[0].split())

    s1 = sim_jacc(stringa1[1].split(), stringa2[1].split())

    s2 = sim_hamming(stringa1[2].split(), stringa2[2].split())
    s3 = sim_cos(stringa1[3].split(), stringa2[3].split())
    s4 = sim_cos(stringa1[4].split(), stringa2[4].split())
    s5 = sim_lev(stringa1[5].split(), stringa2[5].split())

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_fz2bNoSplit(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = sim_cos(stringa1[0], stringa2[0])

    s1 = sim_jacc(stringa1[1], stringa2[1])

    s2 = sim_hamming(stringa1[2], stringa2[2])
    s3 = sim_cos(stringa1[3], stringa2[3])
    s4 = sim_cos(stringa1[4], stringa2[4])
    s5 = sim_lev(stringa1[5], stringa2[5])

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]
    rm_min = min(vect)
    vect.remove(rm_min)
    rm_max = max(vect)
    vect.remove(rm_max)
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_fz2bNoSplitNoMin(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = sim_cos(stringa1[0], stringa2[0])

    s1 = sim_jacc(stringa1[1], stringa2[1])

    s2 = sim_hamming(stringa1[2], stringa2[2])
    s3 = sim_cos(stringa1[3], stringa2[3])
    s4 = sim_cos(stringa1[4], stringa2[4])
    s5 = sim_lev(stringa1[5], stringa2[5])

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]

    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_fz_boh(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = sim_cos(stringa1[0].split(), stringa2[0].split())

    s1 = sim_sodi(stringa1[1].split(), stringa2[1].split()) #or sodi_s

    s2 = sim_cos(stringa1[2], stringa2[2]) #or cos_s
    s3 = sim_cos(stringa1[3].split(), stringa2[3].split()) # or cos_s
    s4 = sim_sodi(stringa1[4], stringa2[4]) #sodi
    s5 = sim_lev(stringa1[5], stringa2[5])

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]

    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_fz_boh2(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = sim_jacc(stringa1[0].split(), stringa2[0].split())

    s1 = sim_jacc(stringa1[1].split(), stringa2[1].split())

    s2 = sim_hamming(stringa1[2], stringa2[2])
    s3 = sim_jacc(stringa1[3], stringa2[3])
    s4 = jaro(stringa1[4], stringa2[4])
    s5 = sim_lev(stringa1[5], stringa2[5])

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]

    aver = round(sum(vect) / len(vect), 2)
    return [aver]
'''

'''
def sim_bf2_ab(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_sodi(t1s, t2s)[0]*0.3788014897846522 + sim_cos(t1s, t2s)[0]*0.3411662864327195 + sim_jacc(t1s, t2s)[0]*0.28003222378262843
    '''s0 = sim_sodi(t1s, t2s)[0] * 0.119 + sim_cos(t1s, t2s)[0] * 0.116 \
         + sim_jacc(t1s, t2s)[0] * 0.093 + sim_lcs(t1s, t2s)[0] * 0.072 \
         + sim_lev(t1s, t2s)[0] * 0.07 + sim_ngram(t1s, t2s)[0] * 0.07 \
         + sim_jacc(t1, t2)[0] * 0.052 + sim_sbert(t1s, t2s)[0] * 0.045 \
         + sim_sodi(t1, t2)[0] * 0.044 + sim_cos(t1, t2)[0] * 0.044 \
         + sim_hamming(t1s, t2s)[0] * 0.037 + sim_lev(t1, t2)[0] * 0.034 \
         + jaro(t1, t2)[0] * 0.022 + sim_hamming(t1, t2)[0] * 0.022 \
         + sim_bert(t1s, t2s)[0] * 0.021 + sim_ngram(t1, t2)[0] * 0.017 \
         + sim_sbert2(t1s, t2s)[0] * 0.0144 + sim_sbert(t1, t2)[0] * 0.010'''

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_cos(t1s, t2s)[0]*0.37717431759570813 + sim_sodi(t1s, t2s)[0]*0.3619633052969103 + sim_jacc(t1s, t2s)[0]*0.26086237710738164
    '''s1 = sim_cos(t1s, t2s)[0] * 0.138 + sim_sodi(t1s, t2s)[0] * 0.113 + sim_cos(t1, t2)[0] * 0.084 \
         + sim_jacc(t1s, t2s)[0] * 0.084 + sim_lev(t1s, t2s)[0] * 0.064 \
         + sim_ngram(t1s, t2s)[0] * 0.063 + sim_sbert2(t1s, t2s)[0] * 0.057 \
         + sim_lcs(t1s, t2s)[0] * 0.057 + sim_hamming(t1s, t2s)[0] * 0.057 \
         + sim_sbert(t1s, t2s)[0] * 0.051 + sim_sbert2(t1, t2)[0] * 0.039 \
         + sim_bert(t1, t2)[0] * 0.027 + sim_ngram(t1, t2)[0] * 0.0202 + sim_hamming(t1, t2)[0] * 0.0183 \
         + jaro(t1s, t2s)[0] * 0.0176 + sim_lev(t1, t2)[0] * 0.017 + sim_lcs(t1, t2)[0] * 0.016 \
         + sim_jacc(t1, t2)[0] * 0.015 + jaro(t1, t2)[0] * 0.015 + sim_bert(t1s, t2s)[0] * 0.01'''

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = sim_sodi(t1s, t2s)[0]*0.3338552330482958 + sim_jacc(t1s, t2s)[0]*0.3338552330482958 + sim_jacc(t1, t2)[0]*0.3322895339034085
    '''s2 = sim_sbert(t1, t2)[0] * 0.068 + sim_cos(t1, t2)[0] * 0.060 + sim_sodi(t1s, t2s)[0] * 0.058 \
         + sim_jacc(t1s, t2s)[0] * 0.058 + sim_sbert(t1s, t2s)[0] * 0.056 + sim_sbert2(t1, t2)[0] * 0.054 \
         + sim_ngram(t1s, t2s)[0] * 0.054 + sim_sbert2(t1s, t2s)[0] * 0.052 + sim_jacc(t1, t2)[0] * 0.048 \
         + jaro(t1s, t2s)[0] * 0.047 + sim_ngram(t1, t2)[0] * 0.046 + sim_lcs(t1, t2)[0] * 0.043 + sim_lcs(
        t1s, t2s)[0] * 0.043 \
         + sim_hamming(t1s, t2s)[0] * 0.042 + jaro(t1, t2)[0] * 0.038 + sim_bert(t1s, t2s)[0] * 0.037 \
         + sim_bert(t1, t2)[0] * 0.037 + sim_hamming(t1, t2)[0] * 0.036 + sim_lev(t1s, t2s)[0] * 0.035 + sim_lev(t1, t2)[0] * 0.031'''

    vect = [s0, s1, s2]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf2_dds(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_cos(t1s, t2s)[0]*0.57 + sim_sodi(t1s, t2s)[0]*0.43

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_hamming(t1s, t2s)[0]*0.55 + sim_ngram(t1s, t2s)[0]*0.45

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = sim_jacc(t1, t2)[0]*0.51 + sim_lcs(t1, t2)[0]*0.49

    t1 = stringa1[3]
    t2 = stringa2[3]
    t1s = t1.split()
    t2s = t2.split()
    s3 = sim_sodi(t1s, t2s)[0] * 0.5 + sim_jacc(t1s, t2s)[0] * 0.5

    vect = [s0, s1, s2, s3]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf2_dwa(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_cos(t1s, t2s)[0]*0.53 + sim_sodi(t1s, t2s)[0]*0.46

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_ngram(t1s, t2s)[0]*0.51 + sim_hamming(t1s, t2s)[0]*0.49

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = sim_hamming(t1s, t2s)[0]*0.55 + sim_jacc(t1, t2)[0]*0.45

    t1 = stringa1[3]
    t2 = stringa2[3]
    t1s = t1.split()
    t2s = t2.split()
    s3 = sim_ngram(t1s, t2s)[0] * 0.62 + sim_cos(t1, t2)[0] * 0.37

    t1 = stringa1[4]
    t2 = stringa2[4]
    t1s = t1.split()
    t2s = t2.split()
    s4 = sim_ngram(t1s, t2s)[0] * 0.5 + sim_sodi(t1s, t2s)[0] * 0.5

    vect = [s0, s1, s2, s3, s4]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf2_dai(stringa1, stringa2):
    t1 = stringa1[0]
    t2 = stringa2[0]
    t1s = t1.split()
    t2s = t2.split()
    s0 = sim_sodi(t1s, t2s)[0]*0.56 + sim_jacc(t1s, t2s)[0]*0.44

    t1 = stringa1[1]
    t2 = stringa2[1]
    t1s = t1.split()
    t2s = t2.split()
    s1 = sim_jacc(t1s, t2s)[0]*0.53 + sim_ngram(t1s, t2s)[0]*0.47

    t1 = stringa1[2]
    t2 = stringa2[2]
    t1s = t1.split()
    t2s = t2.split()
    s2 = sim_hamming(t1s, t2s)[0]*0.53 + sim_jacc(t1, t2)[0]*0.47

    t1 = stringa1[3]
    t2 = stringa2[3]
    t1s = t1.split()
    t2s = t2.split()
    s3 = sim_cos(t1s, t2s)[0] * 0.58 + sim_jacc(t1s, t2s)[0] * 0.42

    vect = [s0, s1, s2, s3]
    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_fz_ok(stringa1, stringa2):
    # jaro, jacc, cos, lev, jac, lev
    s0 = sim_cos(stringa1[0].split(), stringa2[0].split())

    s1 = sim_cos(stringa1[1].split(), stringa2[1].split()) #or sodi_s

    s2 = sim_jacc(stringa1[2], stringa2[2]) #or cos_s
    s3 = sim_cos(stringa1[3].split(), stringa2[3].split()) # or cos_s
    s4 = jaro(stringa1[4], stringa2[4]) #sodi
    s5 = sim_lev(stringa1[5], stringa2[5])

    vect = [s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]]

    aver = round(sum(vect) / len(vect), 2)
    return [aver]

def sim_bf_ag(stringa1, stringa2):

    s0 = sim_cos(stringa1[0], stringa2[0])

    s1 = sim_cos(stringa1[1], stringa2[1])

    s2 = sim_ngram(stringa1[2], stringa2[2])
    s3 = sim_hamming(stringa1[3], stringa2[3])

    vect = [s0[0], s1[0], s2[0], s3[0]]

    aver = round(sum(vect) / len(vect), 2)
    # print(aver)
    return [aver]

def min_cos(data):
    cosine = []
    for el in data:
        stringa1 = concatenate_list_data(el[0])
        stringa2 = concatenate_list_data(el[1])
        cos_sim = get_cosine(text_to_vector(stringa1), text_to_vector(stringa2))
        cosine.append(cos_sim)

    return min(cosine)

def sim_bert(stringa1, stringa2):
    _, _, e1 = extract_bert(' '.join(stringa1), tokenizer, model)
    _, _, e2 = extract_bert(' '.join(stringa2), tokenizer, model)

    a = torch.mean(e1, 0)
    a[torch.isnan(a)] = 0
    b = torch.mean(e2, 0)
    b[torch.isnan(b)] = 0

    v1 = numpy.nan_to_num(a.numpy())
    v2 = numpy.nan_to_num(b.numpy())
    distance = numpy.nan_to_num(scipy.spatial.distance.cosine(v1, v2))
    return [1 - distance]

def sim_sbert(stringa1, stringa2):
    e1 = embedder.encode([' '.join(stringa1)])
    e1 = numpy.nan_to_num(e1)
    e2 = embedder.encode([' '.join(stringa2)])
    e2 = numpy.nan_to_num(e2)
    return [1 - scipy.spatial.distance.cosine(e1, e2)]

def sim_sbert2(stringa1, stringa2):
    e1 = encoder.encode([' '.join(stringa1)])
    e1 = numpy.nan_to_num(e1)
    e2 = encoder.encode([' '.join(stringa2)])
    e2 = numpy.nan_to_num(e2)
    return [1 - scipy.spatial.distance.cosine(e1, e2)]

def extract_bert(text, tokenizer, model):
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]

    n_chunks = int(numpy.ceil(float(text_ids.size(1)) / 510))
    states = []

    for ci in range(n_chunks):
        try:
            text_ids_ = text_ids[0, 1 + ci * 510:1 + (ci + 1) * 510]
            torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
            if text_ids[0, -1] != text_ids[0, -1]:
                torch.cat([text_ids, text_ids[0, -1].unsqueeze(0)])

            with torch.no_grad():
                state = model(text_ids_.unsqueeze(0))[0]
                state = state[:, 1:-1, :]
            states.append(state)
        except:
            pass
    state = torch.cat(states, axis=1)
    return text_ids, text_words, state[0]

from transformers import *
from sentence_transformers import SentenceTransformer
from sentence_transformers import models


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
model = AutoModel.from_pretrained('bert-base-uncased')
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

PATH = "/home/tteofili/Downloads/"
model.save_pretrained(PATH)
tokenizer.save_pretrained(PATH)
embedding = models.BERT(PATH, max_seq_length=128,do_lower_case=True)
pooling_model = models.Pooling(embedding.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model2 = SentenceTransformer(modules=[embedding, pooling_model])
model2.save(PATH)
encoder = SentenceTransformer(PATH)
