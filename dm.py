#!/usr/bin/env python
# coding: utf-8

import deepmatcher as dm

# read dataset

trainLab, validationLab, testLab = dm.data.process(path='/Users/tommasoteofili/Downloads/beer_datasetDM/',
                                                   train='trainLab.csv', validation='validLab.csv', test='testLab.csv')


f1 = 0
runs = 10
for i in range(runs):

    # initialize default deep matcher model
    model = dm.MatchingModel()
    model.initialize(trainLab)

    print("--> TRAIN BASE MODEL <--")
    # train default model with standard dataset
    model.run_train(trainLab, validationLab, best_save_path='best_default_model.pth', epochs=15)

    print("--> EVALUATE BASE MODEL <--")
    # eval default model on test set
    f1 += model.run_eval(testLab)

print('test set avg f1:' + str(f1 / runs))
