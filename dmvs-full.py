#!/usr/bin/env python
# coding: utf-8

import torch
import deepmatcher as dm
import collections

# read datasets

trainLab, validationLab, testLab = dm.data.process(path='/Users/tommasoteofili/Downloads/beer_datasetDM/',
                                                   train='trainLab.csv', validation='validLab.csv', test='testLab.csv')

trainSIM, validationSIM, testSIM = dm.data.process(path='/Users/tommasoteofili/Downloads/beer_datasetDM/',
                                                   train='trainSIM.csv', validation='validSIM.csv', test='testSIM.csv')

f1 = 0
runs = 10
for i in range(runs):

    highway = dm.modules.Transform('2-layer-highway', hidden_size=300, output_size=300)

    # define similarity layer module
    similarity_layer = torch.nn.Sequential(collections.OrderedDict([
        ('highway', highway),
        ('sigmoid', dm.modules.Transform('1-layer-sigmoid', input_size=300, output_size=1))
    ]))

    # create new deep matcher model with similarity module (instead of default classifier)
    pretrained_model = dm.MatchingModel(classifier=lambda: similarity_layer)
    pretrained_model.initialize(trainSIM)

    print("--> PRETRAIN VINSIM MODEL <--")
    # pretrain vinsim model using similarity dataset
    pretrained_model.run_train(trainSIM, validationSIM, best_save_path='best_pretrained_model.pth',
                               criterion=torch.nn.MSELoss(), epochs=15)

    # create softmax layer as per dm original model
    softmax_layer = torch.nn.Sequential(collections.OrderedDict([
        ('combine', dm.modules.Transform('1-layer', non_linearity=None, output_size=2)),
        ('logsoftmax', torch.nn.LogSoftmax(dim=1))
    ]))

    #highway_layer = torch.nn.Sequential(*list(pretrained_model.classifier.children())[:-1])

    # create updated classifier with highway layers and softmax layer
    updated_classifier = torch.nn.Sequential(collections.OrderedDict([
        ('highway', highway),
        ('logsoftmax', softmax_layer)
    ]))

    # attach default deep matcher classifier to pretrained model
    pretrained_model.classifier = updated_classifier

    run_iter = dm.data.MatchingIterator(
        trainLab,
        trainLab,
        train=False,
        batch_size=4,
        device='cpu',
        sort_in_buckets=False)
    init_batch = next(run_iter.__iter__())
    pretrained_model.forward(init_batch)

    print(pretrained_model)

    print("--> FINETUNE PRETRAINED MODEL <--")
    # fine tune pretrained model with standard dataset
    pretrained_model.run_train(trainLab, validationLab, best_save_path='best_finetuned_model.pth', epochs=15)

    # eval fine tuned model on test set
    print("--> EVALUATE FINETUNED MODEL <--")
    f1 += pretrained_model.run_eval(testLab)

print('test set avg f1:' + str(f1 / runs))
