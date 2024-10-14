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
from fairness_training import pretrain_epochs_with_resampled_ensemble_sst_reg_eval_unfairness_valid_partial_safe
from collaborative_models import matrixFactorization, sst_pred

from tqdm import tqdm

parser = argparse.ArgumentParser(description='fairRec')
parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='7',
                        help="device id to run")
parser.add_argument("--beta", 
                    type = float,
                    default = 0.001,
                    help = "Beta in KL-Closed form solution.")
parser.add_argument("--embed_size", type=int, default= 64, help= "the embedding size of MF")
parser.add_argument("--output_size", type=int, default= 1, help="the output size of MF")
parser.add_argument("--num_epochs", type=int, default= 200, help= "the max epoch of training")
parser.add_argument("--learning_rate", type= float, default= 1e-3, help="the learning rate for MF model")
parser.add_argument("--batch_size", type= int, default= 32768, help= "the batchsize for training")
parser.add_argument("--evaluation_epoch", type= int, default= 1, help= "the evaluation epoch")
parser.add_argument("--weight_decay", type= float, default= 1e-5, help= "the weight_decay for training")
parser.add_argument("--top_K", type=int, default= 5, help="the NDCG evaluation @ K")
parser.add_argument('--seed', type=int, default=1, help="the random seed")
parser.add_argument("--saving_path", type=str, default= "./org_MF_temp", help= "the saving path for model")
parser.add_argument("--result_csv", type=str, default="./orig_MF_temp/result_contrast.csv", help="the path for saving result")
parser.add_argument("--data_path", type=str, default="./datasets/ml-1m/", help= "the data path")
parser.add_argument("--fair_reg", type=float, default= 1e-2, help= "the regulator for fairness")
parser.add_argument("--partial_ratio_male", type=float, default= 0.5, help= "the known ratio for training sensitive attr male ")
parser.add_argument("--partial_ratio_female", type=float, default= 0.3, help= "the known ratio for training sensitive attr female ")
parser.add_argument("--orig_unfair_model", type=str, default= "./pretrained_model/ml-1m/MF_orig_model")
parser.add_argument("--gender_train_epoch", type=int, default= 1000, help="the epoch for gender classifier training")
parser.add_argument("--task_type",type=str,default="ml-1m",help="Specify task type: ml-1m/tenrec/lastfm-1K/lastfm-360K")

args = parser.parse_args()
import os,sys 
cur_path = os.getcwd()
path_to_dataset = os.path.join(cur_path,"datasets",args.task_type)
assert os.path.exists(path_to_dataset)
train_csv_path = os.path.join(path_to_dataset,"train.csv")
valid_csv_path = os.path.join(path_to_dataset,"valid.csv")
test_csv_path = os.path.join(path_to_dataset,"test.csv")
sensitive_csv_path = os.path.join(path_to_dataset,"sensitive_attribute.csv")
sensitive_csv_random_path = os.path.join(path_to_dataset,"sensitive_attribute_random.csv")


#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = args.seed
set_random_seed(RANDOM_STATE)

device = torch.device("cuda:"+(args.gpu_id))
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
beta = args.beta 
# random_samples = 100
top_K = args.top_K



data_path = args.data_path
train_data = pd.read_csv(train_csv_path,dtype=np.int64)
valid_data = pd.read_csv(valid_csv_path,dtype=np.int64)
test_data = pd.read_csv(test_csv_path,dtype=np.int64)
orig_sensitive_attr = pd.read_csv(sensitive_csv_path,dtype=np.int64)
sensitive_attr = pd.read_csv(sensitive_csv_random_path,dtype=np.int64)
gender_known_male =  sensitive_attr[sensitive_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_male * sum(sensitive_attr["gender"] == 0))]
gender_known_female =  sensitive_attr[sensitive_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_female * sum(sensitive_attr["gender"] == 1))]


num_uniqueUsers = max(train_data.user_id) + 1
num_uniqueLikes = max(train_data.item_id) + 1
MF_model = matrixFactorization(np.int64(num_uniqueUsers), np.int64(num_uniqueLikes), emb_size).to(device)
print(args)

# the range of priors used in Multiple Prior Guided Robust Optimization
# here we choose 37 different priors
resample_range =[ "0.1", "0.105","0.11","0.12", "0.125", "0.13", "0.14", "0.15", "0.17","0.18", "0.2", "0.22", "0.25", "0.29", "0.33", "0.4", "0.5", "0.67", "1.0",
                  "1.5", "2.0", "2.5","3.0","3.5","4.0","4.5","5.0","5.5", "6.0","6.5", "7.0","7.5","8.0","8.5","9.0","9.5","10.0",
                  ]
# for same resample ratio, we use 3 different random seeds to avoid running into extreme cases.
seed_range = [1, 2, 3]
male_ratio = args.partial_ratio_male 
female_ratio = args.partial_ratio_female 
predicted_sensitive_attr_dict = {}
main_dir = os.path.join("./predict_sst", f"{args.task_type}_{male_ratio}_male_{female_ratio}_female_gender_train_epoch_1000")

for resample_ratio in resample_range:
    predicted_sensitive_attr_dict[resample_ratio] = {}
    for seed_sample in seed_range:
      predicted_sensitive_attr_dict[resample_ratio][seed_sample] = pd.read_csv(os.path.join(main_dir, f"resample_{resample_ratio}_seed{seed_sample}.csv"))



best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, unfairness_val, unfairness_test, best_epoch, best_model = \
        pretrain_epochs_with_resampled_ensemble_sst_reg_eval_unfairness_valid_partial_safe(MF_model, train_data,num_epochs,learning_rate, weight_decay, batch_size, beta, valid_data, \
            test_data, predicted_sensitive_attr_dict, orig_sensitive_attr, top_K, fair_reg ,gender_known_male, gender_known_female, device = device, evaluation_epoch= evaluation_epoch, unsqueeze=True)

os.makedirs(args.saving_path, exist_ok= True)
torch.save(MF_model.state_dict(), args.saving_path + "/MF_model")
torch.save(best_model.state_dict(), args.saving_path + "/best_model")

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
# evaluate the model

# evaluate the model

# avg_HR_preTrain, avg_NDCG_preTrain = evaluate_model(MF_model,test_data.values,top_K, num_uniqueLikes)


# np.savetxt(saving_path + '/avg_HR_preTrain.txt', avg_HR_preTrain)
# np.savetxt(saving_path + '/avg_NDCG_preTrain.txt', avg_NDCG_preTrain)

#sys.stdout.close()