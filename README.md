# Fair Recommendation with Biased Limited Sensitive Attribute

# Dataset
You can download the original dataset from the following links:
[ml-1m](https://grouplens.org/datasets/movielens/1m/)
[Lastfm-360K](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html), we also provide the processed dataset in the folder ./datasets/

## 1. Pretrain MF model

on **ml-1m** 
```bash
bash ./scripts/pretrain/pretrain_ml_1m.sh
```
on **Lastfm-360K**
```bash
bash ./scripts/pretrain/pretrain_Lastfm_360K.sh
```
We also provide the pretrained **checkpoints** in the folder ./pretrained_model, you can also train your own and put it in the folder.


## 2. [Optional] Verify bias exists 
on **ml-1m**  
```bash
bash ./scripts/contrast/change_ratio_and_epoch/partial_ratio_male0.5/run_ml_1m.sh
```
on **Lastfm-360K**
```bash 
bash ./scripts/contrast/change_ratio_and_epoch/partial_ratio_male0.5/run_Lastfm_360K.sh
```

## 3. Create Multiple priority
We first establish a predefined set of prior distributions $\mathcal{P}$, which in our setting is $\{ 1/10.0, 1/9.5, 1/9.0, \cdots, 1/1.5, 1, 1.5, 2, \cdots, 9.5, 10 \}$
We then estimate the distribution of users’ sensitive attributes under each prior distribution $\hat{p}_0 \in \mathcal{P}$ by resampling the known sensitive
attributes.

on **ml-1m**  
```bash
bash ./scripts/predict_sst/run_ml_1m.sh
```
on **Lastfm-360K**
```bash 
bash ./scripts/predict_sst/run_Lastfm_360K.sh
```


## 4. Robust Fairness Optimization 

We have a hyper-parameter $\beta$ in our optimization objective.
In general, as $\beta$ decreases, the fairness metrics improve.

on **ml-1m**  
```bash
bash ./scripts/MPR/male_0.5/run_ml_1m_safe.sh
```
on **Lastfm-360K**
```bash 
bash ./scripts/MPR/male_0.5/run_Lastfm_360K_safe.sh
```
