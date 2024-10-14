gpu_id=0
partial_ratio_male=0.5
gender_train_epoch=1000
main_folder=./predict_sst/
task_type=Lastfm-360K
orig_unfair_model=./pretrained_model/${task_type}/MF_orig_model
for partial_ratio_female in 0.05 0.1 0.2 0.3 0.4 0.5
do 
for resample_ratio in 0.105 0.12 0.125 0.13 0.15 0.18 0.22 0.29 0.33 0.40 0.67 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 1 2 3 4 5 6 7 8 9 10 0.1 0.11 0.13 0.14 0.17 0.2 0.25
do 
for seed in  1 2 3
do 
mkdir -p ${main_folder}${task_type}_${partial_ratio_male}_male_${partial_ratio_female}_female_gender_train_epoch_${gender_train_epoch}
nohup python3 /data/zhangjz/shenhy/MF_fairness/predict_sst.py --gpu_id=${gpu_id} --seed=${seed} --partial_ratio_male=${partial_ratio_male} --data_path="./datasets/${task_type}/" \
 --orig_unfair_model="./pretrained_model/${task_type}/MF_orig_model" --task_type ${task_type} --orig_unfair_model=${orig_unfair_model}\
 --partial_ratio_female=${partial_ratio_female} --gender_train_epoch=${gender_train_epoch} --prior_male_female_ratio_resample=${resample_ratio} \
> ${main_folder}${task_type}_${partial_ratio_male}_male_${partial_ratio_female}_female_gender_train_epoch_${gender_train_epoch}/train_resample_${resample_ratio}_seed${seed}.log
done 
done 
done 