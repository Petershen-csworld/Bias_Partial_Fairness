import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.metrics import accuracy_score 


# model evaluation: hit rate and NDCG

def evaluate_model(model,df_val,top_K, device):
    model.eval()
    avg_UAUC = np.zeros((max(df_val["user_id"].unique()) + 1))
    avg_NDCG = np.zeros((max(df_val["user_id"].unique()) + 1))
    uniq_count= 0
    
    for i in df_val["user_id"].unique():
        test_user_item_rating = df_val[df_val["user_id"]==i]
        test_user_input = torch.tensor(np.array(test_user_item_rating["user_id"])).to(device)
        test_item_input = torch.tensor(np.array(test_user_item_rating["item_id"])).to(device)
        test_rating = np.array(test_user_item_rating["label"])

        y_hat = model(test_user_input, test_item_input)
        y_hat = y_hat.cpu().detach().numpy()
        if len(np.unique(test_rating)) != 1:
            avg_NDCG[i] = ndcg_score(test_rating.reshape((1,-1)), y_hat.reshape((1,-1)), k=top_K)
            avg_UAUC[i] = roc_auc_score(test_rating, y_hat)
            uniq_count += 1

    avg_NDCG = np.sum(avg_NDCG) / uniq_count
    avg_UAUC = np.sum(avg_UAUC) / uniq_count
    return avg_UAUC, avg_NDCG 

def calc_naive_gender_unfairness(model, df_val, df_sensitive_attr, device):
    model.eval()
    rating_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
    for i in df_val["user_id"].unique():
        test_user_item_rating = df_val[df_val["user_id"]==i]
        test_user_input = torch.tensor(np.array(test_user_item_rating["user_id"])).to(device)
        test_item_input = torch.tensor(np.array(test_user_item_rating["item_id"])).to(device)
        # test_rating = np.array(test_user_item_rating["label"])
        test_user_sst = int(df_sensitive_attr[i:i+1]["gender"])

        y_hat = model(test_user_input, test_item_input)
        y_hat = y_hat.cpu().detach().numpy().tolist()
        rating_dict[test_user_sst] += y_hat
    naive_gender_unfairness = np.abs(np.mean(rating_dict[1]) - np.mean(rating_dict[0]))
    return naive_gender_unfairness


def evaluate_model_performance_and_naive_fairness(model, df_val, df_sensitive_attr, top_K, device):
    model.eval()
    avg_UAUC = np.zeros((max(df_val["user_id"].unique())+ 1))
    avg_NDCG = np.zeros((max(df_val["user_id"].unique())+ 1))
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"])).to(device)
        test_item_total = torch.tensor(np.array(df_val["item_id"])).to(device)
        pred_total = model(test_user_total, test_item_total)
        pred_total = pred_total.cpu().detach().numpy()
        uniq_count= 0
        naive_fairness_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
        df_val_with_index = df_val.reset_index()
        group_val = df_val_with_index.groupby("user_id")
        df_sensitive_dict = df_sensitive_attr.set_index("user_id")
        for name, group in tqdm(group_val):
            test_rating = np.array(group["label"])
            y_hat = pred_total[group.index]
            if len(np.unique(test_rating)) != 1:
                avg_NDCG[name] = ndcg_score(test_rating.reshape((1,-1)), y_hat.reshape((1,-1)), k=top_K)
                avg_UAUC[name] = roc_auc_score(test_rating, y_hat)
                uniq_count += 1
            gender = int(df_sensitive_dict.iloc[name]["gender"])
            naive_fairness_dict[gender] += y_hat.tolist()

        avg_NDCG = np.sum(avg_NDCG) / uniq_count
        avg_UAUC = np.sum(avg_UAUC) / uniq_count
        naive_gender_unfairness = float(np.abs(np.mean(naive_fairness_dict[1]) - (np.mean(naive_fairness_dict[0]))))
    return round(avg_UAUC,4), round(avg_NDCG,4), naive_gender_unfairness

def precision(label_list,top_K):
    return sum(label_list[:top_K])/top_K
def recall(label_list,top_K):
    return sum(label_list)[:top_K]/sum(label_list) 
def f_score(label_list,top_K,beta):
    p,r = precision(label_list,top_K), recall(label_list,top_K)
    return (1 + beta**2) * p * r / (beta ** 2) * p + r 

def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2**label_list[i] - 1)/np.log2(i+2)
        dcgsum += dcg
    return dcgsum

def NDCG(label_list,top_n):
    if top_n==None:
        dcg = DCG(label_list)
        ideal_list = sorted(label_list, reverse=True)
        ideal_dcg = DCG(ideal_list)
        if ideal_dcg == 0:
            return 0
        return dcg/ideal_dcg
    else:
        dcg = DCG(label_list[0:top_n])
        ideal_list = sorted(label_list, reverse=True)
        ideal_dcg = DCG(ideal_list[0:top_n])
        if ideal_dcg == 0:
            return 0
        return dcg/ideal_dcg

def calAUC(label_list):
    rank = label_list[::-1]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    posNum = sum(rank)
    negNum = len(rank) - sum(rank)
    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    return auc

def Recall(label_list, top_n):
    top_list = label_list[0:top_n]
    return sum(top_list)/sum(label_list)


# prob = y_hat
# labels = test_rating

def evaluate_model_performance_and_naive_fairness_fast(model, df_val, df_sensitive_attr, top_K, device):
    model.eval()
    avg_UAUC = np.zeros((max(df_val["user_id"].unique())+ 1))
    avg_NDCG = np.zeros((max(df_val["user_id"].unique())+ 1))
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"])).to(device)
        test_item_total = torch.tensor(np.array(df_val["item_id"])).to(device)
        pred_total = model(test_user_total, test_item_total)
        pred_total = pred_total.cpu().detach()
        uniq_count= 0
        naive_fairness_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
        df_val_with_index = df_val.reset_index()
        group_val = df_val_with_index.groupby("user_id")
        df_sensitive_dict = df_sensitive_attr.set_index("user_id")
        for name, group in group_val:
            test_rating = np.array(group["label"]).astype(int)
            y_hat = pred_total[group.index] 
            if len(np.unique(test_rating)) != 1:
                # avg_NDCG[name] = ndcg_score(test_rating.reshape((1,-1)), y_hat.reshape((1,-1)), k=top_K)
                y_hat_sort_id = y_hat.sort(descending=True).indices
                label_rank = test_rating[y_hat_sort_id]
                avg_NDCG[name] = NDCG(label_rank, top_K)
                avg_UAUC[name] = calAUC(label_rank)
                uniq_count += 1
            gender = int(df_sensitive_dict.iloc[name]["gender"])
            naive_fairness_dict[gender] += y_hat.tolist()

        avg_NDCG = np.sum(avg_NDCG) / uniq_count
        avg_UAUC = np.sum(avg_UAUC) / uniq_count
        naive_gender_unfairness = float(np.abs(np.mean(naive_fairness_dict[1]) - (np.mean(naive_fairness_dict[0]))))
    return round(avg_UAUC,4), round(avg_NDCG,4), naive_gender_unfairness


def evaluate_model_performance_and_naive_fairness_EO_fast(model, df_val, df_sensitive_attr, top_K, device):
    model.eval()
    avg_UAUC = np.zeros((max(df_val["user_id"].unique()) + 1))
    avg_NDCG = np.zeros((max(df_val["user_id"].unique()) + 1))
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"])).to(device)
        test_item_total = torch.tensor(np.array(df_val["item_id"])).to(device)
        pred_total = model(test_user_total, test_item_total)
        pred_total = pred_total.cpu().detach()
        uniq_count= 0
        naive_fairness_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
        df_val_with_index = df_val.reset_index()
        group_val = df_val_with_index.groupby("user_id")
        df_sensitive_dict = df_sensitive_attr.set_index("user_id")
        for name, group in group_val:
            test_rating = np.array(group["label"]).astype(int)
            y_hat = pred_total[group.index]
            if len(np.unique(test_rating)) != 1:
                # avg_NDCG[name] = ndcg_score(test_rating.reshape((1,-1)), y_hat.reshape((1,-1)), k=top_K)
                y_hat_sort_id = y_hat.sort(descending=True).indices
                label_rank = test_rating[y_hat_sort_id]
                avg_NDCG[name] = NDCG(label_rank, top_K)
                avg_UAUC[name] = calAUC(label_rank)
                uniq_count += 1
            gender = int(df_sensitive_dict.iloc[name]["gender"])
            pos_ratings = y_hat[test_rating == 1]
            naive_fairness_dict[gender] += pos_ratings.tolist()

        avg_NDCG = np.sum(avg_NDCG) / uniq_count
        avg_UAUC = np.sum(avg_UAUC) / uniq_count
        naive_gender_unfairness = float(np.abs(np.mean(naive_fairness_dict[1]) - (np.mean(naive_fairness_dict[0]))))
    return round(avg_UAUC,4), round(avg_NDCG,4), naive_gender_unfairness



# evaluation gender classifier
def evaluation_gender(data, label, model):
    model.eval()
    pred = model(data)
    pred_out = pred.argmax(1)
    acc = round(sum(pred_out == label).item()/(pred_out.shape[0]) * 100, 2)
    pred_male_female_ratio = ((sum(pred_out == 0).item() + 1e-2)/(sum(pred_out == 1).item() + 1e-2))
    return acc, pred_male_female_ratio



def evaluate_model_performance_and_naive_fairness_fast_partial_valid(model, df_val, df_sensitive_attr, gender_known_male, gender_known_female, top_K, device):
    model.eval()
    avg_UAUC = np.zeros((max(df_val["user_id"].unique()) + 1))
    avg_NDCG = np.zeros((max(df_val["user_id"].unique()) + 1))
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"])).to(device)
        test_item_total = torch.tensor(np.array(df_val["item_id"])).to(device)
        pred_total = model(test_user_total, test_item_total)
        pred_total = pred_total.cpu().detach()
        uniq_count= 0
        fairness_count = 0
        naive_fairness_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
        df_val_with_index = df_val.reset_index()
        group_val = df_val_with_index.groupby("user_id")
        df_sensitive_dict = df_sensitive_attr.set_index("user_id")
        for name, group in group_val:
            test_rating = np.array(group["label"]).astype(int)
            y_hat = pred_total[group.index]
            if len(np.unique(test_rating)) != 1:
                # avg_NDCG[name] = ndcg_score(test_rating.reshape((1,-1)), y_hat.reshape((1,-1)), k=top_K)
                y_hat_sort_id = y_hat.sort(descending=True).indices
                label_rank = test_rating[y_hat_sort_id]
                avg_NDCG[name] = NDCG(label_rank, top_K)
                avg_UAUC[name] = calAUC(label_rank)
                uniq_count += 1
            if (name in gender_known_male) or (name in gender_known_female): 
                gender = int(df_sensitive_dict.iloc[name]["gender"])
                naive_fairness_dict[gender] += y_hat.tolist()
                fairness_count += 1

        avg_NDCG = np.sum(avg_NDCG) / uniq_count
        avg_UAUC = np.sum(avg_UAUC) / uniq_count
        naive_gender_unfairness = float(np.abs(np.mean(naive_fairness_dict[1]) - (np.mean(naive_fairness_dict[0]))))
    return round(avg_UAUC,4), round(avg_NDCG,4), naive_gender_unfairness

def evaluation_gender_new(data, label, model):
    
    model.eval()
    pred = model(data)
    pred_out = pred.argmax(1)
    gammas = {}
    acc = round(sum(pred_out == label).item()/(pred_out.shape[0]) * 100, 2)
    gammas[0] = 1 - accuracy_score(pred_out[label == 0].cpu().numpy(), label[label == 0].cpu().numpy())
    gammas[1] = 1 - accuracy_score(pred_out[label == 1].cpu().numpy(), label[label == 1].cpu().numpy())
        
    return acc, gammas 