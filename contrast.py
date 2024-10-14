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
                        default='0',
                        help="device id to run")
# parser.add_argument('--saving_path', type=str, default="./result_rerun/", help="the path to save result")
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
parser.add_argument("--data_path", type=str, default="./datasets/Lastfm-360K/", help= "the data path")
parser.add_argument("--fair_reg", type=float, default= 0.1, help= "the regulator for fairness")
parser.add_argument("--partial_ratio_male", type=float, default= 0.5, help= "the known ratio for training sensitive attr male ")
parser.add_argument("--partial_ratio_female", type=float, default= 0.05, help= "the known ratio for training sensitive attr female ")
parser.add_argument("--orig_unfair_model", type=str, default= "./pretrained_model/Lastfm-360K/MF_orig_model")
parser.add_argument("--gender_train_epoch", type=int, default= 100, help="the epoch for gender classifier training")
parser.add_argument("--task_type",type=str,default="Lastfm-36K",help="Specify task type: ml-1m/tenrec/lastfm-1K/lastfm-360K")

args = parser.parse_args()

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
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

# random_samples = 100
top_K = args.top_K



data_path = args.data_path
train_data = pd.read_csv(data_path + "train.csv",dtype=np.int64)
valid_data = pd.read_csv(data_path + "valid.csv",dtype=np.int64)
test_data = pd.read_csv(data_path + "test.csv",dtype=np.int64)
orig_sensitive_attr = pd.read_csv(data_path + "sensitive_attribute.csv",dtype=np.int64)
sensitive_attr = pd.read_csv(data_path + "sensitive_attribute_random.csv",dtype=np.int64)



num_users = len(sensitive_attr)
train_sensitive_attr = sensitive_attr[:np.int64(0.8 * num_users)]
test_sensitive_attr = sensitive_attr[np.int64(0.8 * num_users):] 

if args.task_type == "Lastfm-360K":
  orig_male = 73882 
  orig_female = 20970
elif args.task_type == "ml-1m":
  orig_male = 4331
  orig_female = 1709
orig_male_ratio = orig_male / (orig_male + orig_female)

num_tot_male = train_sensitive_attr[train_sensitive_attr["gender"] == 0].shape[0]
num_tot_female = train_sensitive_attr[train_sensitive_attr["gender"] == 1].shape[0]
tot_sampling = int(num_tot_male * args.partial_ratio_male) + int(num_tot_female * args.partial_ratio_female)


male_num_known = int(tot_sampling * orig_male_ratio )
female_num_known = int(tot_sampling * (1 - orig_male_ratio)) 

gender_known_male = train_sensitive_attr[train_sensitive_attr["gender"] == 0].sample(n = min(male_num_known, num_tot_male),
                                                                                     random_state = args.seed)["user_id"].to_numpy()
gender_known_female = train_sensitive_attr[train_sensitive_attr["gender"] == 1].sample(n = min(female_num_known, num_tot_female),
                                                                                     random_state = args.seed)["user_id"].to_numpy()

"""
We want to make sure that
the ratio in test datasetsatisfies that:
male/female = orig_male_female_ratio
"""
num_male_test = test_sensitive_attr[test_sensitive_attr["gender"] == 0].shape[0]
num_female_test = test_sensitive_attr[test_sensitive_attr["gender"] == 1].shape[0]
num_tot_test = int(num_male_test * args.partial_ratio_male) + int(num_female_test * args.partial_ratio_female)
male_sample_test_num  =  min(int(num_tot_test * orig_male_ratio) ,num_male_test)
female_sample_test_num = min(int(num_tot_test * (1 - orig_male_ratio)) ,num_female_test)

gender_known_male_test =  test_sensitive_attr[test_sensitive_attr["gender"] == 0]["user_id"].to_numpy()[: male_sample_test_num]
gender_known_female_test =  test_sensitive_attr[test_sensitive_attr["gender"] == 1]["user_id"].to_numpy()[: female_sample_test_num]

orig_model = torch.load(args.orig_unfair_model, map_location = torch.device("cpu"))
user_embedding = orig_model['user_emb.weight']
user_embedding = user_embedding.detach().to(device)
classifier_model = sst_pred(user_embedding.shape[1], 32, 2).to(device)

"""
Note: here a similar approach is adopted, besides, the test set below is not for NCF training, but for sensitive
attribute reconstruction. 
"""
sensitive_num = sensitive_attr.shape[0]
sampled_sensitive = int(orig_male * args.partial_ratio_male) + int(orig_female * args.partial_ratio_female)

sensitive_attr_reshuffled = sensitive_attr.sample(frac=1,
                                                  random_state=args.seed).reset_index(drop=True)
test_known_male = sensitive_attr_reshuffled[sensitive_attr_reshuffled["gender"] == 0]["user_id"].to_numpy()[: min(int(sampled_sensitive * orig_male_ratio),orig_male)]
test_known_female = sensitive_attr_reshuffled[sensitive_attr_reshuffled["gender"] == 1]["user_id"].to_numpy()[: min(int(sampled_sensitive * (1 - orig_male_ratio)),orig_female)]
test_tensor = torch.cat([user_embedding[test_known_male], user_embedding[test_known_female]])
test_label = torch.cat([torch.zeros(test_known_male.shape[0]), torch.ones(test_known_female.shape[0])]).to(device)


test_tensor_unseen = torch.cat([user_embedding[gender_known_male_test], user_embedding[gender_known_female_test]])
test_label_unseen = torch.cat([torch.zeros(gender_known_male_test.shape[0]), torch.ones(gender_known_female_test.shape[0])]).to(device)

train_tensor = torch.cat([user_embedding[gender_known_male], user_embedding[gender_known_female]])
train_label = torch.cat([torch.zeros(gender_known_male.shape[0]), torch.ones(gender_known_female.shape[0])]).to(device)
print(f"Original male_female number and ratio : {orig_male},{orig_female},{orig_male/orig_female}")
print(f"Train male_female number and ratio : {gender_known_male.shape[0]},{gender_known_female.shape[0]},{gender_known_male.shape[0]/gender_known_female.shape[0]}")
print(f"Test male_female number and ratio : {test_known_male.shape[0]},{test_known_female.shape[0]},{test_known_male.shape[0]/test_known_female.shape[0]}")
print(f"Test Unseen male_female number and ratio : {gender_known_male_test.shape[0]},{gender_known_female_test.shape[0]},{gender_known_male_test.shape[0]/gender_known_female_test.shape[0]}")


optimizer_for_classifier = torch.optim.Adam(classifier_model.parameters(), lr=1e-2, weight_decay= 1e-5)
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



orig_gender_known_male =  sensitive_attr[sensitive_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_male * sum(sensitive_attr["gender"] == 0))]
orig_gender_known_female =  sensitive_attr[sensitive_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_female * sum(sensitive_attr["gender"] == 1))]

pred_all_label = classifier_model(user_embedding).max(1).indices

pred_all_label[orig_gender_known_male] = 0
pred_all_label[orig_gender_known_female] = 1

pred_sensitive_attr = pd.DataFrame(list(zip(list(range(len(sensitive_attr))), list(pred_all_label.cpu().tolist()))),\
     columns = ["user_id", "gender"])

# sensitive_attr
# test_pred_female = sum(classifier_model(test_tensor).max(1).indices)
# test_pred_male = len(classifier_model(test_tensor).max(1).indices) - test_pred_female

# construct predicted test 

num_uniqueUsers = max(train_data.user_id) + 1
# num_uniqueLikes = len(train_data.like_id.unique())
num_uniqueLikes = max(train_data.item_id) + 1

# start training the NCF model

MF_model = matrixFactorization(num_uniqueUsers, num_uniqueLikes, emb_size).to(device)
print(args)
best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, unfairness_val, unfairness_test, best_epoch, best_model = \
        pretrain_epochs_with_predicted_sst_reg_eval_unfairness_valid_partial(MF_model,train_data,num_epochs,learning_rate, weight_decay, batch_size, valid_data, \
            test_data, pred_sensitive_attr, orig_sensitive_attr, top_K, fair_reg ,gender_known_male, gender_known_female, device, evaluation_epoch= evaluation_epoch, unsqueeze=True)

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
        writer.writerow(["args", "best_val_ndcg", "val_UAUC_in_that_epoch", "test_ndcg_in_that_epoch", "test_UAUC_in_that_epoch", "unfairness_val_partial", "unfairness_test", "best_epoch", "sst_test_male_female_ratio", "sst_test_acc","unseen_test_acc","sst_unseen_test_male_female_ratio"])

with open(args.result_csv,"a") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow([args, best_val_ndcg, val_UAUC_in_that_epoch, test_ndcg_in_that_epoch, test_UAUC_in_that_epoch, unfairness_val, unfairness_test, best_epoch, test_pred_male_female_ratio, test_acc, test_unseen_acc, test_pred_male_female_ratio_unseen])
