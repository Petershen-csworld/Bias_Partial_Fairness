# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, ndcg_score, recall_score
from evaluation import evaluate_model, evaluate_model_performance_and_naive_fairness_fast, evaluation_gender
from sklearn.model_selection import train_test_split 
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
from fairness_training import pretrain_epochs_with_predicted_sst_reg_eval_unfairness_valid_partial
from collaborative_models import matrixFactorization, sst_pred

from tqdm import tqdm

parser = argparse.ArgumentParser(description='fairRec')
parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='5',
                        help="device id to run")

parser.add_argument('--seed', type=int, default=1, help="the random seed")
parser.add_argument("--saving_path", type=str, default= "./predict_sst", help= "the saving path for model")
parser.add_argument("--data_path", type=str, default="./datasets/Lastfm-360K/", help= "the data path")
parser.add_argument("--fair_reg", type=float, default= 0.1, help= "the regulator for fairness")
parser.add_argument("--partial_ratio_male", type=float, default= 0.5, help= "the known ratio for training sensitive attr male ")
parser.add_argument("--partial_ratio_female", type=float, default= 0.2, help= "the known ratio for training sensitive attr female ")
parser.add_argument("--orig_unfair_model", type=str, default= "./pretrained_model/Lastfm-360K/MF_orig_model")
parser.add_argument("--gender_train_epoch", type=int, default= 1, help="the epoch for gender classifier training")
parser.add_argument("--prior_male_female_ratio_resample", type=float, default=0.5, help="the prior resample ratio for male")
parser.add_argument("--task_type",type=str,default="Lastfm-360K",help="Specify task type: ml-1m/tenrec/lastfm-1K/lastfm-360K")

args = parser.parse_args()

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = args.seed
set_random_seed(1)

device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
# set hyperparameters
saving_path = args.saving_path
# random_samples = 100
data_path = args.data_path
orig_sensitive_attr = pd.read_csv(data_path + "sensitive_attribute.csv",dtype=np.int64)
sensitive_attr = pd.read_csv(data_path + "sensitive_attribute_random.csv",dtype=np.int64)




#80% for training in user
num_users = len(sensitive_attr)
train_sensitive_attr = sensitive_attr[:np.int64(0.8 * num_users)]
test_sensitive_attr = sensitive_attr[np.int64(0.8 * num_users):] 

# generating sensitive attr mask for training

gender_known_male =  train_sensitive_attr[train_sensitive_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_male * sum(train_sensitive_attr["gender"] == 0))]
gender_known_female =  train_sensitive_attr[train_sensitive_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_female * sum(train_sensitive_attr["gender"] == 1))]
# sensitive_attr["gender"]
# args.partial_ratio_male


gender_known_male_test =  test_sensitive_attr[test_sensitive_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_male * sum(test_sensitive_attr["gender"] == 0))]
gender_known_female_test =  test_sensitive_attr[test_sensitive_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_female * sum(test_sensitive_attr["gender"] == 1))]

orig_model = torch.load(args.orig_unfair_model, map_location = torch.device("cpu"))
user_embedding = orig_model['user_emb.weight']
user_embedding = user_embedding.detach().to(device)
classifier_model = sst_pred(user_embedding.shape[1], 32, 2).to(device)

# no validation set

# construct test set
# 一个测试集:全部样本作为测试集

sensitive_attr_reshuffled = sensitive_attr.sample(frac=1).reset_index(drop=True)
test_known_male = sensitive_attr_reshuffled[sensitive_attr_reshuffled["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_male * sum(sensitive_attr_reshuffled["gender"] == 0))]
test_known_female = sensitive_attr_reshuffled[sensitive_attr_reshuffled["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_female * sum(sensitive_attr_reshuffled["gender"] == 1))]
test_tensor = torch.cat([user_embedding[test_known_male], user_embedding[test_known_female]])
test_label = torch.cat([torch.zeros(test_known_male.shape[0]), torch.ones(test_known_female.shape[0])]).to(device)

# 另一个测试集:未见过的20%用户

test_tensor_unseen = torch.cat([user_embedding[gender_known_male_test], user_embedding[gender_known_female_test]])
test_label_unseen = torch.cat([torch.zeros(gender_known_male_test.shape[0]), torch.ones(gender_known_female_test.shape[0])]).to(device)



# resample male
# 如果根据oracle 的样本数量采样大于已知样本，就直接将已知的所有样本放进去

np.random.seed(args.seed)
if int(len(gender_known_female) * args.prior_male_female_ratio_resample) < len(gender_known_male):
    resampled_known_male = np.random.choice(gender_known_male, int(len(gender_known_female) * args.prior_male_female_ratio_resample), replace=False)
else:
    resampled_known_male = copy.deepcopy(gender_known_male)

train_tensor = torch.cat([user_embedding[resampled_known_male], user_embedding[gender_known_female]])
train_label = torch.cat([torch.zeros(resampled_known_male.shape[0]), torch.ones(gender_known_female.shape[0])]).to(device)

optimizer_for_classifier = torch.optim.Adam(classifier_model.parameters(), lr=1e-3, weight_decay= 1e-5)
loss_for_classifier = torch.nn.CrossEntropyLoss()

for i in range(args.gender_train_epoch):
    train_pred = classifier_model(train_tensor)
    loss_train = loss_for_classifier(train_pred, train_label.type(torch.LongTensor).to(device))
    optimizer_for_classifier.zero_grad()
    loss_train.backward()
    optimizer_for_classifier.step()
    # print("loss train:" + str(loss_train.item()))
    train_acc, train_pred_male_female_ratio = evaluation_gender(train_tensor, train_label, classifier_model)
    test_acc, test_pred_male_female_ratio = evaluation_gender(test_tensor, test_label, classifier_model)
    test_unseen_acc, test_pred_male_female_ratio_unseen = evaluation_gender(test_tensor_unseen,test_label_unseen,classifier_model)

    

print("test acc on unseen 20%_user:" + str(test_unseen_acc))
print("test acc:" + str(test_acc))
print("test_20%_unseen_pred_male_female_ratio:" + str(test_pred_male_female_ratio_unseen))
print("test_pred_male_female_ratio:" + str(test_pred_male_female_ratio))



gender_known_male =  sensitive_attr[sensitive_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_male * sum(sensitive_attr["gender"] == 0))]
gender_known_female =  sensitive_attr[sensitive_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_female * sum(sensitive_attr["gender"] == 1))]

pred_all_label = classifier_model(user_embedding).max(1).indices

# 将已知的敏感属性复制回去
pred_all_label[gender_known_male] = 0
pred_all_label[gender_known_female] = 1

pred_sensitive_attr = pd.DataFrame(list(zip(list(range(len(sensitive_attr))), list(pred_all_label.cpu().tolist()))),\
     columns = ["user_id", "gender"])


subdir = f"{args.task_type}_{args.partial_ratio_male}_male_{args.partial_ratio_female}_female_gender_train_epoch_{args.gender_train_epoch}"
os.makedirs(os.path.join(args.saving_path, subdir), exist_ok = True)
save_csv = f"resample_{args.prior_male_female_ratio_resample}_seed{args.seed}.csv"
pred_sensitive_attr.to_csv(os.path.join(args.saving_path, subdir, save_csv), index = None)
