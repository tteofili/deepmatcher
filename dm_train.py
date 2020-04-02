import deepmatcher as dm
import pandas as pd
import torch
import collections

datadir = '/home/tteofili/Downloads/dataset/custom/'


def join(train1, valid1, test1, train2, valid2, name):
    names = []
    names.append('id')
    names.append('label')
    attr_per_tab = int((len(train1[0]) - 2) / 2)
    for i in range(attr_per_tab):
        names.append('left_attr_' + str(i))
    for i in range(attr_per_tab):
        names.append('right_attr_' + str(i))

    joined = pd.concat(
        [pd.DataFrame(train1), pd.DataFrame(valid1), pd.DataFrame(test1), pd.DataFrame(train2), pd.DataFrame(valid2)],
        ignore_index=True, sort=False)
    joined.columns = names

    joined.to_csv(datadir + name + '/all.csv', index=False)
    all = dm.data.process(path=datadir + name, train='all.csv')

    return all


def finetune_dm(all_train, pretrained_model, trainLab, validationLab):
    # initialize default deepmatcher model
    base_model = dm.MatchingModel()
    base_model.initialize(all_train)
    updated_classifier = base_model.classifier

    # attach default deep matcher classifier to pretrained model
    pretrained_model.classifier = updated_classifier
    print(pretrained_model)

    print("FINETUNING with "+str(len(trainLab))+" samples")
    # fine tune pretrained model with standard dataset
    pretrained_model.run_train(trainLab, validationLab, best_save_path='best_finetuned_model.pth', epochs=15, optimizer=dm.optim.Optimizer(lr=0.0001))
    return pretrained_model


def pretrain_dm(all_train, name, train, valid, cut):

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
    print(df.head())
    df.to_csv(datadir + name + '/trainSim_' + str(cut) + '.csv', index=False)

    df = pd.DataFrame(valid)
    df.columns = names
    print(df.head())
    df.to_csv(datadir + name + '/validSim_' + str(cut) + '.csv', index=False)

    # read dataset
    trainSIM, validationSIM = dm.data.process(path=datadir + name,
                                              train='trainSim_' + str(cut) + '.csv',
                                              validation='validSim_' + str(cut) + '.csv')

    # define similarity layer module
    similarity_layer = torch.nn.Sequential(collections.OrderedDict([
        ('highway', dm.modules.Transform('2-layer-highway', hidden_size=300, output_size=300)),
        ('sigmoid', dm.modules.Transform('1-layer-sigmoid', input_size=300, output_size=1))
    ]))

    # create new deep matcher model with similarity module (instead of default classifier)
    pretrained_model = dm.MatchingModel(classifier=lambda: similarity_layer)

    pretrained_model.initialize(all_train)

    print("PRETRAINING with "+str(len(train))+" samples")
    # pretrain vinsim model using similarity dataset
    pretrained_model.run_train(trainSIM, validationSIM, best_save_path='best_pretrained_model.pth',
                               criterion=torch.nn.MSELoss(), epochs=15, optimizer=dm.optim.Optimizer(lr=0.1))

    return pretrained_model


def train_dm(name, train, valid, test, cut):
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

    # initialize default deep matcher model
    model = dm.MatchingModel()
    model.initialize(trainLab)

    print("TRAINING with "+str(len(train))+" samples")
    # train default model with standard dataset
    model.run_train(trainLab, validationLab, best_save_path='best_default_model.pth', epochs=15)

    return model, trainLab, validationLab, testLab


def eval_dm(model, data):
    return model.run_eval(data)

def pt_ft_dm_full(all_train, name, sim_train, sim_valid, trainLab, validationLab):
    names = []
    names.append('id')
    names.append('label')
    attr_per_tab = int((len(sim_train[0]) - 2) / 2)
    for i in range(attr_per_tab):
        names.append('left_attr_' + str(i))
    for i in range(attr_per_tab):
        names.append('right_attr_' + str(i))

    df = pd.DataFrame(sim_train)
    df.columns = names
    print(df.head())
    df.to_csv(datadir + name + '/trainSim.csv', index=False)

    df = pd.DataFrame(sim_valid)
    df.columns = names
    print(df.head())
    df.to_csv(datadir + name + '/validSim.csv', index=False)

    # read dataset
    trainSIM, validationSIM = dm.data.process(path=datadir + name,
                                              train='trainSim.csv',
                                              validation='validSim.csv')

    highway = dm.modules.Transform('2-layer-highway', hidden_size=300, output_size=300)

    # define similarity layer module
    similarity_layer = torch.nn.Sequential(collections.OrderedDict([
        ('highway', highway),
        ('sigmoid', dm.modules.Transform('1-layer-sigmoid', input_size=300, output_size=1))
    ]))

    # create new deep matcher model with similarity module (instead of default classifier)
    pretrained_model = dm.MatchingModel(classifier=lambda: similarity_layer)
    pretrained_model.initialize(all_train)

    print("--> PRETRAIN VINSIM MODEL <--")
    # pretrain vinsim model using similarity dataset
    pretrained_model.run_train(trainSIM, validationSIM, best_save_path='best_pretrained_model.pth',
                               criterion=torch.nn.MSELoss(), epochs=15)

    # create softmax layer as per dm original model
    softmax_layer = torch.nn.Sequential(collections.OrderedDict([
        ('combine', dm.modules.Transform('1-layer', non_linearity=None, output_size=2)),
        ('logsoftmax', torch.nn.LogSoftmax(dim=1))
    ]))

    # highway_layer = torch.nn.Sequential(*list(pretrained_model.classifier.children())[:-1])

    # create updated classifier with highway layers and softmax layer
    updated_classifier = torch.nn.Sequential(collections.OrderedDict([
        ('highway', highway),
        ('logsoftmax', softmax_layer)
    ]))

    # attach default deep matcher classifier to pretrained model
    pretrained_model.classifier = updated_classifier

    '''run_iter = dm.data.MatchingIterator(
        trainLab,
        trainLab,
        train=False,
        batch_size=4,
        device='cpu',
        sort_in_buckets=False)
    init_batch = next(run_iter.__iter__())
    pretrained_model.forward(init_batch)
    '''
    print(pretrained_model)

    print("--> FINETUNE PRETRAINED MODEL <--")
    # fine tune pretrained model with standard dataset
    pretrained_model.run_train(trainLab, validationLab, best_save_path='best_finetuned_model.pth', epochs=15)
    return pretrained_model

def pt_ft_dm_classifier(all_train, name, sim_train, sim_valid, trainLab, validationLab):
    names = []
    names.append('id')
    names.append('label')
    attr_per_tab = int((len(sim_train[0]) - 2) / 2)
    for i in range(attr_per_tab):
        names.append('left_attr_' + str(i))
    for i in range(attr_per_tab):
        names.append('right_attr_' + str(i))

    df = pd.DataFrame(sim_train)
    df.columns = names
    print(df.head())
    df.to_csv(datadir + name + '/trainSim.csv', index=False)

    df = pd.DataFrame(sim_valid)
    df.columns = names
    print(df.head())
    df.to_csv(datadir + name + '/validSim.csv', index=False)

    # read dataset
    trainSIM, validationSIM = dm.data.process(path=datadir + name,
                                              train='trainSim.csv',
                                              validation='validSim.csv')

    # define similarity layer module
    similarity_layer = torch.nn.Sequential(collections.OrderedDict([
        ('highway', dm.modules.Transform('2-layer-highway', hidden_size=300, output_size=300)),
        ('sigmoid', dm.modules.Transform('1-layer-sigmoid', input_size=300, output_size=1))
    ]))

    # create new deep matcher model with similarity module (instead of default classifier)
    pretrained_model = dm.MatchingModel(classifier=lambda: similarity_layer)

    pretrained_model.initialize(all_train)

    print("PRETRAINING with "+str(len(trainSIM))+" samples")
    # pretrain vinsim model using similarity dataset
    pretrained_model.run_train(trainSIM, validationSIM, best_save_path='best_pretrained_model.pth',
                               criterion=torch.nn.BCELoss(), epochs=15, optimizer=dm.optim.Optimizer(lr=0.001), batch_size=32)

    # initialize default deepmatcher model
    base_model = dm.MatchingModel()
    base_model.initialize(all_train)
    updated_classifier = base_model.classifier

    # attach default deep matcher classifier to pretrained model
    pretrained_model.classifier = updated_classifier
    print(pretrained_model)

    print("FINETUNING with " + str(len(trainLab)) + " samples")
    # fine tune pretrained model with standard dataset
    pretrained_model.run_train(trainLab, validationLab, best_save_path='best_finetuned_model.pth', epochs=15,
                               optimizer=dm.optim.Optimizer(lr=0.001))
    return pretrained_model