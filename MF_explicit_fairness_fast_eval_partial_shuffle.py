# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, ndcg_score, recall_score
from evaluation import evaluate_model, evaluate_model_performance_and_naive_fairness_fast, evaluate_model_performance_and_naive_fairness_fast_partial_valid

import os

import math
import heapq # for retrieval topK
import random
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from collaborative_models import matrixFactorization

from tqdm import tqdm

parser = argparse.ArgumentParser(description='fairRec')
parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
parser.add_argument("--embed_size", type=int, default= 64, help= "the embedding size of MF")
parser.add_argument("--output_size", type=int, default= 1, help="the output size of MF")
parser.add_argument("--num_epochs", type=int, default= 200, help= "the max epoch of training")
parser.add_argument("--learning_rate", type= float, default= 1e-3, help="the learning rate for MF model")
parser.add_argument("--batch_size", type= int, default= 32768, help= "the batchsize for training")
parser.add_argument("--evaluation_epoch", type= int, default= 1, help= "the evaluation epoch")
parser.add_argument("--weight_decay", type= float, default= 1e-7, help= "the weight_decay for training")
parser.add_argument("--top_K", type=int, default= 5, help="the NDCG evaluation @ K")
parser.add_argument('--seed', type=int, default=1, help="the random seed")
parser.add_argument("--saving_path", type=str, default= "./orig_MF_temp", help= "the saving path for model")
parser.add_argument("--result_csv", type=str, default="./orig_MF_temp/result.csv", help="the path for saving result")
parser.add_argument("--data_path", type=str, default="./datasets/Lastfm-360K/", help= "the data path")
parser.add_argument("--fair_reg", type=float, default= 0.1, help= "the regulator for fairness")
parser.add_argument("--partial_ratio_male", type=float, default= 0.1, help= "the known ratio for training sensitive attr male ")
parser.add_argument("--partial_ratio_female", type=float, default= 0.1, help= "the known ratio for training sensitive attr female ")
parser.add_argument("--task_type",type=str,default="Lastfm-360K",help="Specify task type: ml-1m/tenrec/Lastfm(Lastfm-1K)/Lastfm-360K")


args = parser.parse_args()

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = args.seed
set_random_seed(RANDOM_STATE)
device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
# set hyperparameters
saving_path = args.saving_path
emb_size = args.embed_size
output_size = args.output_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
evaluation_epoch = args.evaluation_epoch
weight_decay = args.weight_decay
fair_reg = args.fair_reg
task_type = args.task_type 
# random_samples = 100
top_K = args.top_K

def pretrain_epochs_partial_fairness_reg_eval_unfairness_valid_partial(model, df_train, epochs, lr, weight_decay, batch_size, valid_data, test_data, sensitive_attr, top_K, fair_reg, gender_known_male, gender_known_female, evaluation_epoch=10, unsqueeze=False, shuffle=True):
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
    for idx in range(epochs):
        j = 0
        loss_total = 0
        fair_reg_total = 0
        random_id = np.array([i for i in range(len(df_train))])
        if shuffle:
            np.random.shuffle(random_id)
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size):
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
                 
            # fairness regulation
            # partial female average pred:
            known_female = np.isin(data_batch["user_id"], gender_known_female)
            know_female_pred_mean = y_hat[known_female].mean()

            # partial male average pred:
            known_male = np.isin(data_batch["user_id"], gender_known_male)

            know_male_pred_mean = y_hat[known_male].mean()
            
            # if no male or female, then the regulation is set to 0
            if sum(known_female) * sum(known_male) != 0:
                fair_regulation = torch.abs(know_female_pred_mean - know_male_pred_mean) * fair_reg
            else:
                fair_regulation = torch.tensor(0)

            fair_reg_total += fair_regulation.item()
            loss = loss + fair_regulation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j = j+1
        print('epoch: ', idx, 'average loss: ',loss_total/ j, "fair reg:", fair_reg_total/j)
        if idx % evaluation_epoch == 0 :
            t0 = time.time()
            avg_UAUC_val, avg_NDCG_val, naive_unfairness_val = evaluate_model_performance_and_naive_fairness_fast_partial_valid(model, valid_data, sensitive_attr, gender_known_male, gender_known_female, top_K, device)
            t1 = time.time()
            avg_UAUC_test, avg_NDCG_test, naive_unfairness_test = evaluate_model_performance_and_naive_fairness_fast(model, test_data, sensitive_attr, top_K, device)
            t2 = time.time()
            print('epoch: ', idx, 'validation NDCG@' + str(top_K) + ':' ,avg_NDCG_val, 'UAUC:' ,avg_UAUC_val, 'Unfairness:', naive_unfairness_val, "time:", t1 - t0)
            print('epoch: ', idx, 'test NDCG@' + str(top_K) + ':' ,avg_NDCG_test, 'UAUC:' ,avg_UAUC_test, "Unfairness:", naive_unfairness_test, "time:", t2 - t1)

            if avg_NDCG_val > best_val_ndcg:
                best_val_ndcg = avg_NDCG_val
                val_UAUC_in_that_epoch = avg_UAUC_val
                test_ndcg_in_that_epoch = avg_NDCG_test
                test_UAUC_in_that_epoch = avg_UAUC_test
                best_epoch = idx
                naive_unfairness_val_in_that_epoch = naive_unfairness_val
                naive_unfairness_test_in_that_epoch = naive_unfairness_test

    return best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, naive_unfairness_val_in_that_epoch, naive_unfairness_test_in_that_epoch, best_epoch


# load data
# model = MF_model
# df_val = valid_data
# df_sensitive_attr = sensitive_attr

data_path = args.data_path
train_data = pd.read_csv(data_path + "train.csv",dtype=np.int64)
valid_data = pd.read_csv(data_path + "valid.csv",dtype=np.int64)
test_data = pd.read_csv(data_path + "test.csv",dtype=np.int64)
sensitive_attr = pd.read_csv(data_path + "sensitive_attribute.csv",dtype=np.int64)
random_gender_attr = pd.read_csv(data_path + "sensitive_attribute_random.csv",dtype=np.int64)

# generating sensitive attr mask for training
# gender_known_male =  sensitive_attr[sensitive_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_male * sum(sensitive_attr["gender"] == 0))]
# gender_known_female =  sensitive_attr[sensitive_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_female * sum(sensitive_attr["gender"] == 1))]
# sensitive_attr["gender"]
# args.partial_ratio_male

# generating sensitive attr mask from shuffled gender list
gender_known_male =  random_gender_attr[random_gender_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_male * sum(random_gender_attr["gender"] == 0))]
gender_known_female =  random_gender_attr[random_gender_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_female * sum(random_gender_attr["gender"] == 1))]


num_uniqueUsers = max(train_data.user_id) + 1
# num_uniqueLikes = len(train_data.like_id.unique())
num_uniqueLikes = max(train_data.item_id) + 1
# start training the NCF model
print(int(num_uniqueLikes))
print(int(num_uniqueUsers))
MF_model = matrixFactorization(np.int64(num_uniqueUsers), np.int64(num_uniqueLikes), emb_size).to(device)


# model = MF_model
# df_train = train_data
# epochs = num_epochs
# lr = learning_rate
# batch_size = batch_size
# # num_negatives = num_negatives
# unsqueeze=True

print(args)



best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, unfairness_val, unfairness_test, best_epoch = \
        pretrain_epochs_partial_fairness_reg_eval_unfairness_valid_partial(MF_model,train_data,num_epochs,learning_rate, weight_decay, batch_size, valid_data, \
            test_data, sensitive_attr, top_K, fair_reg ,gender_known_male, gender_known_female, evaluation_epoch= evaluation_epoch, unsqueeze=True)

os.makedirs(args.saving_path, exist_ok= True)
torch.save(MF_model.state_dict(), args.saving_path + "/MF_orig_model")

csv_folder = ''
for path in args.result_csv.split("/")[:-1]:
    csv_folder = os.path.join(csv_folder, path)

os.makedirs(csv_folder, exist_ok= True)

try:
    pd.read_csv(args.result_csv)
except:
    with open(args.result_csv,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["args", "best_val_ndcg", "val_UAUC_in_that_epoch", "test_ndcg_in_that_epoch", "test_UAUC_in_that_epoch", "unfairness_val_partial", "unfairness_test", "best_epoch"])

with open(args.result_csv,"a") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow([args, best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, unfairness_val, unfairness_test, best_epoch])
