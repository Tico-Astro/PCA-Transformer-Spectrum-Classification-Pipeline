import math
import numpy as np
import random
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from moudel.mgmcformer import MgMcFORMER

warnings.filterwarnings("ignore")

import numpy as np
import torch

'''
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
'''

class EarlyStoppingForLabel0:
    def __init__(self, patience=10, min_delta=0.0, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf  # 这里是分数越高越好
        self.early_stop = False
        self.path = path

    def __call__(self, label0_score, model):
        # 比较的是分数是否有提升
        if label0_score > self.best_score + self.min_delta:
            self.best_score = label0_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)




class FocalLoss(nn.Module):
    def __init__(self,gamma=2,weight=None):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self,inputs,targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma*ce_loss
        return focal_loss

# loading user-specified hyperparameters
def get_user_specified_hyperparameters(args):
    prop = {}
    prop['batch'], prop['lr'], prop['nlayers'], prop['emb_size'], prop['nhead'], prop['masking_ratio'], prop[
        'emb_size_c'] = \
        args.batch, args.lr, args.nlayers, args.emb_size, args.nhead, args.masking_ratio, args.emb_size_c
    return prop


# loading fixed hyperparameters
def get_fixed_hyperparameters(prop, args):
    prop['epochs'], prop['ratio_highest_attention'] = args.epochs, args.ratio_highest_attention
    prop['dropout'], prop['nhid'], prop['nhid_c'], prop['dataset'], prop[
        'multi_group'] = args.dropout, args.nhid, args.nhid_c, args.dataset, args.multi_group
    return prop


def get_prop(args):
    prop = get_user_specified_hyperparameters(args)
    prop = get_fixed_hyperparameters(prop, args)
    return prop


def data_loader(dataset):
    # load data
    dir = './data/' + dataset + '/'
    X_train = np.load(dir + 'X_train.npy')
    X_test = np.load(dir + 'X_test.npy')
    y_train = np.load(dir + 'y_train.npy')
    y_test = np.load(dir + 'y_test.npy')

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return X_train, y_train, X_test, y_test


def mean_standardize_fit(X):
    m1 = np.mean(X, axis=1)
    mean = np.mean(m1, axis=0)

    s1 = np.std(X, axis=1)
    std = np.mean(s1, axis=0)

    return mean, std


def mean_standardize_transform(X, mean, std):
    return (X - mean) / std


def preprocess(X_train, y_train, X_test, y_test, prop):
    mean, std = mean_standardize_fit(X_train)
    X_train, X_test = mean_standardize_transform(X_train, mean, std), mean_standardize_transform(X_test, mean, std)

    X_train_task = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()
    y_train_task = torch.as_tensor(y_train)
    y_test = torch.as_tensor(y_test)

    return X_train_task, y_train_task, X_test, y_test,[mean,std]


def preprocess_1(X_train, y_train, X_test, y_test, prop, ls):
    mean, std = ls
    X_train, X_test = mean_standardize_transform(X_train, mean, std), mean_standardize_transform(X_test, mean, std)

    X_train_task = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()
    y_train_task = torch.as_tensor(y_train)
    y_test = torch.as_tensor(y_test)

    return X_train_task, y_train_task, X_test, y_test


def initialize_training(prop):
    model = MgMcFORMER(prop['multi_group'], prop['nclasses'], prop['seq_len'], prop['input_size'], prop['emb_size'], \
                       prop['nhid'], prop['emb_size_c'], prop['nhid_c'], prop['nhead'], prop['nlayers'], prop['device'],
                       prop['dropout']).to(prop['device'])

    #weights = torch.tensor([4,1,2]).to('cuda:0') 
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.CrossEntropyLoss(weight = weights)
    #weights = torch.tensor([1.5, 0.1,0.3,1.5,1.0]).to('cuda:0')
    #criterion = FocalLoss(gamma=2, weight = weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=prop['lr'])

    return model, criterion, optimizer


def attention_sampled_masking_heuristic(masking_ratio, ratio_highest_attention, instance_channel, prop):
    res, index = instance_channel.topk(int(math.ceil(ratio_highest_attention * prop['emb_size'])))
    index = index.cpu().data.tolist()

    index2 = [random.sample(index[i], int(math.ceil(masking_ratio * prop['emb_size']))) for i in
              range(prop['batch_true'])]
    return np.array(index2)


def random_instance_masking(masking_ratio, ratio_highest_attention, instance_channel, prop):
    indices = attention_sampled_masking_heuristic(masking_ratio, ratio_highest_attention, instance_channel, prop)
    boolean_indices = np.array([[True if i in index else False for i in range(prop['emb_size'])] for index in indices])
    boolean_indices_masked = np.repeat(boolean_indices[:, np.newaxis, :], prop['seq_len'], axis=1)

    return boolean_indices_masked


def multitask_train(model, criterion, optimizer, dataloader_train, boolean_indices_masked, prop):
    model.train()
    total_loss = 0
    attn_arr = []

    count = 0
    for data, label in dataloader_train:
        data = data.to(prop['device'])
        label = label.to(prop['device'])
        mask_c = torch.as_tensor(boolean_indices_masked[count * prop['batch']:count * prop['batch'] + len(label)],
                                 device=prop['device'])

        optimizer.zero_grad()
        y_pred, attn = model(data, mask_c)
        loss = criterion(y_pred, label)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        count += 1
        attn_arr.append(torch.sum(attn, axis=1) - torch.diagonal(attn, offset=0, dim1=1, dim2=2))
    instance_channel = torch.cat(attn_arr, dim=0)

    return total_loss, instance_channel


def evaluate(y_pred, y, nclasses, criterion, device):
    results = []

    task_type = 'classification'
    if task_type == 'classification':
        loss = criterion(y_pred, torch.as_tensor(y, device=device)).item()
        pred, target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
        pred = np.argmax(pred, axis=1)

        acc = accuracy_score(target, pred)
        prec = precision_score(target, pred, average='macro')
        rec = recall_score(target, pred, average='macro')
        f1 = f1_score(target, pred, average='macro')

        # 👇 添加 label=0 的 F1-score
        f1_label0 = f1_score(target, pred, labels=[0], average='macro')

        results.extend([loss, acc, prec, rec, f1, f1_label0])

    return results


def test(model, dataloader_test, nclasses, device, criterion):
    model.eval()  # Turn on the evaluation mode

    output_arr = []
    label_arr = []
    with torch.no_grad():
        for data, label in dataloader_test:
            data = data.to(device)
            #print(data)
            label = label.to(device)
            pred = model(data)[0]
            output_arr.append(pred)
            label_arr.append(label)

    return evaluate(torch.cat(output_arr, 0), torch.cat(label_arr, 0), nclasses, criterion, device)

from sklearn.model_selection import KFold

# KFold划分数据集
# 修改后的 create_k_folds
def create_k_folds(X, y, n_splits=5):
    # 创建 KFold 对象
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    for train_idx, val_idx in kf.split(X):
        splits.append((train_idx, val_idx))
    
    return splits


'''
def training(model, optimizer, criterion, dataloader_train, dataloader_test, prop):
    task_loss_arr = []
    acc, acc_arr = 0, []
    test_result_arr = []

    # channel_mask   B*emb_size*emb_size
    instance_channel = torch.as_tensor(torch.rand(prop['batch_true'], prop['emb_size']), device=prop['device'])

    for epoch in range(1, prop['epochs'] + 1):

        boolean_indices_masked = random_instance_masking(prop['masking_ratio'], prop['ratio_highest_attention'],
                                                         instance_channel, prop)
        task_loss, instance_channel = multitask_train(model, criterion, optimizer, dataloader_train,
                                                      boolean_indices_masked, prop)

        task_loss_arr.append(task_loss)

        test_result = test(model, dataloader_test, prop['nclasses'], prop['device'], criterion)
        train_result = test(model, dataloader_train, prop['nclasses'], prop['device'], criterion)

        
        print('Epoch: ' + str(epoch) + ', TASK Loss: ' + str(task_loss)+', TEST loss: '+str(test_result[0])+', TRAIN Acc: '+str(train_result[1]))

        if epoch % 10 == 0:
            train_result = test(model, dataloader_train, prop['nclasses'], prop['device'], criterion)
            print('train acc: ', train_result[1])
            test_result = test(model, dataloader_test, prop['nclasses'], prop['device'], criterion)
            if test_result[1] > acc:
                acc = test_result[1]
            acc_arr.append(test_result[1])
            test_result_arr.append(test_result)
            print('Dataset: ' + prop['dataset'] + ',Max Acc: ', acc, 'Now test Acc', test_result[1])

    torch.save(model.state_dict(), "test_model_ZTF.pth")
    print("Model saved as 'test_model_ZTF.pth'")

    torch.cuda.empty_cache()
'''


def training(model, optimizer, criterion, dataloader_train, dataloader_test, prop):
    task_loss_arr = []
    acc, acc_arr = 0, []
    test_result_arr = []

    # 初始化 early stopping
    # early_stopping = EarlyStopping(patience=99, path='best_model_ZTF.pt')
    early_stopping = EarlyStoppingForLabel0(patience=99, path='best_model_ZTF.pt')

    instance_channel = torch.as_tensor(torch.rand(prop['batch_true'], prop['emb_size']), device=prop['device'])

    for epoch in range(1, prop['epochs'] + 1):

        boolean_indices_masked = random_instance_masking(
            prop['masking_ratio'], prop['ratio_highest_attention'],
            instance_channel, prop
        )

        task_loss, instance_channel = multitask_train(
            model, criterion, optimizer, dataloader_train,
            boolean_indices_masked, prop
        )

        task_loss_arr.append(task_loss)

        test_result = test(model, dataloader_test, prop['nclasses'], prop['device'], criterion)
        train_result = test(model, dataloader_train, prop['nclasses'], prop['device'], criterion)

        print(f'Epoch: {epoch}, TASK Loss: {task_loss:.4f}, TEST loss: {test_result[0]:.4f}, '
      f'TRAIN Acc: {train_result[1]:.4f}, TEST Acc: {test_result[1]:.4f}, '
      f'TEST Label-0 F1: {test_result[5]:.4f}')

        # Early stopping 每一轮都进行验证
        early_stopping(test_result[5]+test_result[4], model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

        if epoch % 10 == 0:
            print('train acc: ', train_result[1])
            if test_result[1] > acc:
                acc = test_result[1]
            acc_arr.append(test_result[5])
            test_result_arr.append(test_result)
            print(f'Dataset: {prop["dataset"]}, Max Acc: {acc:.4f}, Now test Acc: {test_result[1]:.4f}')

    # 加载最优模型参数
    model.load_state_dict(torch.load('best_model_ZTF.pt'))
    print("Best model restored from 'best_model_ZTF.pt'")

    torch.save(model.state_dict(), "test_model_ZTF.pth")
    print("Final model saved as 'test_model_ZTF.pth'")

    torch.cuda.empty_cache()


