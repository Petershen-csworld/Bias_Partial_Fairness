import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, ndcg_score, recall_score
from evaluation import evaluate_model, evaluate_model_performance_and_naive_fairness_fast, evaluation_gender, evaluate_model_performance_and_naive_fairness_fast_partial_valid
import random 
import os
import copy
import math
import heapq # for retrieval topK
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import time
from collaborative_models import matrixFactorization, sst_pred

from tqdm import tqdm

# Train the model with fairness regularization 
def pretrain_epochs_with_predicted_sst_reg_eval_unfairness_valid_partial(model, df_train, epochs, lr, weight_decay, batch_size, valid_data, test_data, predicted_sensitive_attr, oracle_sensitive_attr, top_K, fair_reg, gender_known_male, gender_known_female, device, evaluation_epoch=10, unsqueeze=False, shuffle=True):
    criterion = nn.BCELoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # 带L2 正则的优化器
    model.train()
    best_val_ndcg = 0 
    test_ndcg_in_that_epoch = 0
    val_UAUC_in_that_epoch = 0
    test_UAUC_in_that_epoch = 0
    best_epoch = 0
    naive_unfairness_val_in_that_epoch = 0
    naive_unfairness_test_in_that_epoch = 0
    for i in range(epochs):
        j = 0
        loss_total = 0
        fair_reg_total = 0
        random_id = np.array([i for i in range(len(df_train))])
        if shuffle:
            np.random.shuffle(random_id)
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size): # DATA batch
            # data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            data_batch = df_train.loc[random_id[batch_i:(batch_i+batch_size)]].reset_index(drop=True)
            #train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
            # train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_uniqueLikes, num_negatives,device)
            train_ratings = torch.FloatTensor(np.array(data_batch["label"])).to(device)
            train_user_input = torch.LongTensor(np.array(data_batch["user_id"])).to(device)
            train_item_input = torch.LongTensor(np.array(data_batch["item_id"])).to(device)
            train_user_sst =  torch.Tensor(np.array(predicted_sensitive_attr.iloc[np.array(data_batch["user_id"])]["gender"])).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat, train_ratings.view(-1))
            loss_total += loss.item()
            # 公平性正则项 
            # fairness regulation
            fair_regulation = torch.abs((y_hat[train_user_sst == 1]).mean() - (y_hat[train_user_sst == 0]).mean()) * fair_reg
            fair_reg_total += fair_regulation.item()

            loss = loss + fair_regulation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch: ', i, 'batch: ', j, 'out of: ',np.int64(np.floor(len(df_train)/batch_size)), 'average loss: ',loss.item())
            j = j+1
        print('epoch: ', i, 'average loss: ',loss_total/ j, "fair reg:", fair_reg_total/j)
        if i % evaluation_epoch ==0 :
            # avg_UAUC_val, avg_NDCG_val = evaluate_model(model, valid_data, top_K, device)
            # avg_UAUC_test, avg_NDCG_test = evaluate_model(model, test_data, top_K, device)
            # naive_unfairness_val = calc_naive_gender_unfairness(model, valid_data, sensitive_attr, device)
            # naive_unfairness_test = calc_naive_gender_unfairness(model, test_data, sensitive_attr, device)
            t0 = time.time()
            avg_UAUC_val, avg_NDCG_val, naive_unfairness_val = evaluate_model_performance_and_naive_fairness_fast_partial_valid(model, valid_data, oracle_sensitive_attr, gender_known_male, gender_known_female, top_K, device)
            t1 = time.time()
            avg_UAUC_test, avg_NDCG_test, naive_unfairness_test = evaluate_model_performance_and_naive_fairness_fast(model, test_data, oracle_sensitive_attr, top_K, device)
            t2 = time.time()
            print('epoch: ', i, 'validation NDCG@' + str(top_K) + ':' ,avg_NDCG_val, 'UAUC:' ,avg_UAUC_val, 'Partial Valid Unfairness:', naive_unfairness_val, " time:" , str(t1 - t0))
            print('epoch: ', i, 'test NDCG@' + str(top_K) + ':' ,avg_NDCG_test, 'UAUC:' ,avg_UAUC_test, "Unfairness:", naive_unfairness_test, " time:", str(t2 - t1))

            if avg_NDCG_val > best_val_ndcg:
                best_val_ndcg = avg_NDCG_val
                val_UAUC_in_that_epoch = avg_UAUC_val
                test_ndcg_in_that_epoch = avg_NDCG_test
                test_UAUC_in_that_epoch = avg_UAUC_test
                best_epoch = i
                best_model = copy.deepcopy(model)
                naive_unfairness_val_in_that_epoch = naive_unfairness_val
                naive_unfairness_test_in_that_epoch = naive_unfairness_test

    return best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, naive_unfairness_val_in_that_epoch, naive_unfairness_test_in_that_epoch, best_epoch, best_model



def pretrain_epochs_with_resampled_ensemble_sst_reg_eval_unfairness_valid_partial(model, df_train, epochs, lr, weight_decay, batch_size, beta, valid_data, test_data, 
                                                                             predicted_sensitive_attr_dict, oracle_sensitive_attr, top_K, fair_reg, 
                                                                             gender_known_male, gender_known_female, device, 
                                                                             evaluation_epoch=3, unsqueeze=False, shuffle=True):
    # obselete version
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # 带L2 正则的优化器
    model.train()
    best_val_ndcg = 0 
    test_ndcg_in_that_epoch = 0
    val_UAUC_in_that_epoch = 0
    test_UAUC_in_that_epoch = 0
    best_epoch = 0
    naive_unfairness_val_in_that_epoch = 0
    naive_unfairness_test_in_that_epoch = 0
    for i in range(epochs):
        j = 0
        loss_total = 0
        fair_reg_total = 0
        random_id = np.array([i for i in range(len(df_train))])
        if shuffle:
            np.random.shuffle(random_id)
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size): # DATA batch
            # data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            data_batch = df_train.loc[random_id[batch_i:(batch_i+batch_size)]].reset_index(drop=True)
            #train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
            # train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_uniqueLikes, num_negatives,device)
            train_ratings = torch.FloatTensor(np.array(data_batch["label"])).to(device)
            train_user_input = torch.LongTensor(np.array(data_batch["user_id"])).to(device)
            train_item_input = torch.LongTensor(np.array(data_batch["item_id"])).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat, train_ratings.view(-1))
            loss_total += loss.item()
            fair_regulation = torch.Tensor([0.0]).to(device)
            # {resample_ratio: {}}
            for resample_ratio, resample_df in  predicted_sensitive_attr_dict.items():
                 if resample_ratio == "oracle":
                     continue 
                 resampled_user_sst = torch.Tensor(np.array(resample_df.iloc[np.array(data_batch["user_id"])]["gender"])).to(device)
                 resampled_fair_reg = torch.abs((y_hat[ resampled_user_sst == 1]).mean() - (y_hat[ resampled_user_sst == 0]).mean())
                 fair_regulation += torch.exp(resampled_fair_reg/beta)
                                     
            fair_regulation = fair_reg * beta * torch.log(fair_regulation)
            fair_reg_total += fair_regulation.item()
            loss = loss + fair_regulation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch: ', i, 'batch: ', j, 'out of: ',np.int64(np.floor(len(df_train)/batch_size)), 'average loss: ',loss.item())
            j = j+1
        print('epoch: ', i, 'average loss: ',loss_total/ j, "fair reg:", fair_reg_total/j)
        if i % evaluation_epoch ==0 :
            # avg_UAUC_val, avg_NDCG_val = evaluate_model(model, valid_data, top_K, device)
            # avg_UAUC_test, avg_NDCG_test = evaluate_model(model, test_data, top_K, device)
            # naive_unfairness_val = calc_naive_gender_unfairness(model, valid_data, sensitive_attr, device)
            # naive_unfairness_test = calc_naive_gender_unfairness(model, test_data, sensitive_attr, device)
            t0 = time.time()
            avg_UAUC_val, avg_NDCG_val, naive_unfairness_val = evaluate_model_performance_and_naive_fairness_fast_partial_valid(model, valid_data, oracle_sensitive_attr, gender_known_male, gender_known_female, top_K, device)
            t1 = time.time()
            avg_UAUC_test, avg_NDCG_test, naive_unfairness_test = evaluate_model_performance_and_naive_fairness_fast(model, test_data, oracle_sensitive_attr, top_K, device)
            t2 = time.time()
            print('epoch: ', i, 'validation NDCG@' + str(top_K) + ':' ,avg_NDCG_val, 'UAUC:' ,avg_UAUC_val, 'Partial Valid Unfairness:', naive_unfairness_val, " time:" , str(t1 - t0))
            print('epoch: ', i, 'test NDCG@' + str(top_K) + ':' ,avg_NDCG_test, 'UAUC:' ,avg_UAUC_test, "Unfairness:", naive_unfairness_test, " time:", str(t2 - t1))

            if avg_NDCG_val > best_val_ndcg:
                best_val_ndcg = avg_NDCG_val
                val_UAUC_in_that_epoch = avg_UAUC_val
                test_ndcg_in_that_epoch = avg_NDCG_test
                test_UAUC_in_that_epoch = avg_UAUC_test
                best_epoch = i
                best_model = copy.deepcopy(model)
                naive_unfairness_val_in_that_epoch = naive_unfairness_val
                naive_unfairness_test_in_that_epoch = naive_unfairness_test

    return best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, naive_unfairness_val_in_that_epoch, naive_unfairness_test_in_that_epoch, best_epoch, best_model

def pretrain_epochs_with_resampled_ensemble_sst_reg_eval_unfairness_valid_partial_new(model, df_train, epochs, lr, weight_decay, batch_size, beta, valid_data, test_data, 
                                                                             predicted_sensitive_attr_dict, oracle_sensitive_attr, top_K, fair_reg, 
                                                                             gender_known_male, gender_known_female, device, 
                                                                             evaluation_epoch=3, unsqueeze=False, shuffle=True,random_seed=[0,1,2,3]):
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    best_val_ndcg = 0 
    test_ndcg_in_that_epoch = 0
    val_UAUC_in_that_epoch = 0
    test_UAUC_in_that_epoch = 0
    best_epoch = 0
    naive_unfairness_val_in_that_epoch = 0
    naive_unfairness_test_in_that_epoch = 0
    for i in range(epochs):
        j = 0
        loss_total = 0
        fair_reg_total = 0
        random_id = np.array([i for i in range(len(df_train))])
        if shuffle:
            np.random.shuffle(random_id)
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size): # DATA batch
            # data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            data_batch = df_train.loc[random_id[batch_i:(batch_i+batch_size)]].reset_index(drop=True)
            train_ratings = torch.FloatTensor(np.array(data_batch["label"])).to(device)
            train_user_input = torch.LongTensor(np.array(data_batch["user_id"])).to(device)
            train_item_input = torch.LongTensor(np.array(data_batch["item_id"])).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat, train_ratings.view(-1))
            loss_total += loss.item()
            fair_regulation = torch.Tensor([0.0]).to(device)
            # {resample_ratio: {}}
            for resample_ratio, resample_seed_dict in  predicted_sensitive_attr_dict.items():
                for seed in resample_seed_dict.keys():
                 # print(seed)
                 resample_df = resample_seed_dict[seed]
                 resampled_user_sst = torch.Tensor(np.array(resample_df.iloc[np.array(data_batch["user_id"])]["gender"])).to(device)
                 resampled_fair_reg = torch.abs((y_hat[ resampled_user_sst == 1]).mean() - (y_hat[ resampled_user_sst == 0]).mean())
                 fair_regulation += torch.exp(resampled_fair_reg/beta)
                 if False:
                     print(f"Regulation term:[resample_ratio {resample_ratio}] [random seed {random_seed}]: {torch.exp(resampled_fair_reg/beta)}") 
                                
            fair_regulation = fair_reg * beta * torch.log(fair_regulation)
            fair_reg_total += fair_regulation.detach().item()
            # print(fair_regulation)
            loss = loss + fair_regulation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch: ', i, 'batch: ', j, 'out of: ',np.int64(np.floor(len(df_train)/batch_size)), 'average loss: ',loss.item())
            j = j+1
        print('epoch: ', i, 'average loss: ',loss_total/ j, "fair reg:", fair_reg_total/j)
        if i % evaluation_epoch ==0 :
            # avg_UAUC_val, avg_NDCG_val = evaluate_model(model, valid_data, top_K, device)
            # avg_UAUC_test, avg_NDCG_test = evaluate_model(model, test_data, top_K, device)
            # naive_unfairness_val = calc_naive_gender_unfairness(model, valid_data, sensitive_attr, device)
            # naive_unfairness_test = calc_naive_gender_unfairness(model, test_data, sensitive_attr, device)
            t0 = time.time()
            avg_UAUC_val, avg_NDCG_val, naive_unfairness_val = evaluate_model_performance_and_naive_fairness_fast_partial_valid(model, valid_data, oracle_sensitive_attr, gender_known_male, gender_known_female, top_K, device)
            t1 = time.time()
            avg_UAUC_test, avg_NDCG_test, naive_unfairness_test = evaluate_model_performance_and_naive_fairness_fast(model, test_data, oracle_sensitive_attr, top_K, device)
            t2 = time.time()
            print('epoch: ', i, 'validation NDCG@' + str(top_K) + ':' ,avg_NDCG_val, 'UAUC:' ,avg_UAUC_val, 'Partial Valid Unfairness:', naive_unfairness_val, " time:" , str(t1 - t0))
            print('epoch: ', i, 'test NDCG@' + str(top_K) + ':' ,avg_NDCG_test, 'UAUC:' ,avg_UAUC_test, "Unfairness:", naive_unfairness_test, " time:", str(t2 - t1))

            if avg_NDCG_val > best_val_ndcg:
                best_val_ndcg = avg_NDCG_val
                val_UAUC_in_that_epoch = avg_UAUC_val
                test_ndcg_in_that_epoch = avg_NDCG_test
                test_UAUC_in_that_epoch = avg_UAUC_test
                best_epoch = i
                best_model = copy.deepcopy(model)
                naive_unfairness_val_in_that_epoch = naive_unfairness_val
                naive_unfairness_test_in_that_epoch = naive_unfairness_test

    return best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, naive_unfairness_val_in_that_epoch, naive_unfairness_test_in_that_epoch, best_epoch, best_model


def pretrain_epochs_with_resampled_ensemble_sst_reg_eval_unfairness_valid_partial_safe(model, df_train, epochs, lr, weight_decay, batch_size, beta, valid_data, test_data, 
                                                                             predicted_sensitive_attr_dict, oracle_sensitive_attr, top_K, fair_reg, 
                                                                             gender_known_male, gender_known_female, device, 
                                                                             evaluation_epoch=3, unsqueeze=False, shuffle=True,random_seed=[0,1,2,3]):
    # this `safe` version used log sum exp trick to avoid explosion 
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    model.train()
    best_val_ndcg = 0 
    test_ndcg_in_that_epoch = 0
    val_UAUC_in_that_epoch = 0
    test_UAUC_in_that_epoch = 0
    best_epoch = 0
    naive_unfairness_val_in_that_epoch = 0
    naive_unfairness_test_in_that_epoch = 0
    for i in range(epochs):
        j = 0
        loss_total = 0
        fair_reg_total = 0
        random_id = np.array([i for i in range(len(df_train))])
        if shuffle:
            np.random.shuffle(random_id)
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size): # DATA batch
            # data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            data_batch = df_train.loc[random_id[batch_i:(batch_i+batch_size)]].reset_index(drop=True)
            #train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
            # train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_uniqueLikes, num_negatives,device)
            train_ratings = torch.FloatTensor(np.array(data_batch["label"])).to(device)
            train_user_input = torch.LongTensor(np.array(data_batch["user_id"])).to(device)
            train_item_input = torch.LongTensor(np.array(data_batch["item_id"])).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat, train_ratings.view(-1))
            loss_total += loss.item()
            fair_regulation = torch.Tensor([0.0]).to(device)
            C = torch.Tensor([0.0]).to(device)
            reg_dict = {}
            j = 0
            cnt = 0
            for resample_ratio, resample_seed_dict in  predicted_sensitive_attr_dict.items():
                for seed in resample_seed_dict.keys():
                 # print(seed)
                 resample_df = resample_seed_dict[seed]
                 resampled_user_sst = torch.Tensor(np.array(resample_df.iloc[np.array(data_batch["user_id"])]["gender"])).to(device)
                 resampled_fair_reg = torch.abs((y_hat[ resampled_user_sst == 1]).mean() - (y_hat[ resampled_user_sst == 0]).mean())
                 C = torch.max(C, resampled_fair_reg/beta)
                 reg_dict[cnt] = resampled_fair_reg 
             
                 cnt += 1
                 if False:
                     print(f"Regulation term:[resample_ratio {resample_ratio}] [random seed {random_seed}]: {torch.exp(resampled_fair_reg/beta)}") 

            for resample_ratio, reg in reg_dict.items():
                fair_regulation += torch.exp((reg) /beta - C.detach()) # log sum exp trick
            fair_regulation = fair_reg * beta * (torch.log(fair_regulation) + C.detach())
            fair_reg_total += fair_regulation.item()
            loss = loss + fair_regulation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch: ', i, 'batch: ', j, 'out of: ',np.int64(np.floor(len(df_train)/batch_size)), 'average loss: ',loss.item())
            j = j+1
        print('epoch: ', i, 'average loss: ',loss_total/ j, "fair reg:", fair_reg_total/j)
        if i % evaluation_epoch ==0 :
            # avg_UAUC_val, avg_NDCG_val = evaluate_model(model, valid_data, top_K, device)
            # avg_UAUC_test, avg_NDCG_test = evaluate_model(model, test_data, top_K, device)
            # naive_unfairness_val = calc_naive_gender_unfairness(model, valid_data, sensitive_attr, device)
            # naive_unfairness_test = calc_naive_gender_unfairness(model, test_data, sensitive_attr, device)
            t0 = time.time()
            avg_UAUC_val, avg_NDCG_val, naive_unfairness_val = evaluate_model_performance_and_naive_fairness_fast_partial_valid(model, valid_data, oracle_sensitive_attr, gender_known_male, gender_known_female, top_K, device)
            t1 = time.time()
            avg_UAUC_test, avg_NDCG_test, naive_unfairness_test = evaluate_model_performance_and_naive_fairness_fast(model, test_data, oracle_sensitive_attr, top_K, device)
            t2 = time.time()
            print('epoch: ', i, 'validation NDCG@' + str(top_K) + ':' ,avg_NDCG_val, 'UAUC:' ,avg_UAUC_val, 'Partial Valid Unfairness:', naive_unfairness_val, " time:" , str(t1 - t0))
            print('epoch: ', i, 'test NDCG@' + str(top_K) + ':' ,avg_NDCG_test, 'UAUC:' ,avg_UAUC_test, "Unfairness:", naive_unfairness_test, " time:", str(t2 - t1))

            if avg_NDCG_val > best_val_ndcg:
                best_val_ndcg = avg_NDCG_val
                val_UAUC_in_that_epoch = avg_UAUC_val
                test_ndcg_in_that_epoch = avg_NDCG_test
                test_UAUC_in_that_epoch = avg_UAUC_test
                best_epoch = i
                best_model = copy.deepcopy(model)
                naive_unfairness_val_in_that_epoch = naive_unfairness_val
                naive_unfairness_test_in_that_epoch = naive_unfairness_test

    return best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, naive_unfairness_val_in_that_epoch, naive_unfairness_test_in_that_epoch, best_epoch, best_model

