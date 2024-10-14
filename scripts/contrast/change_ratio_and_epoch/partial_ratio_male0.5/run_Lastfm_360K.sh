top_k=5
gpu_id=0
task_type="Lastfm-360K"
for learning_rate in 1e-3
do
for weight_decay in 1e-6
do
for fair_reg in 1e-1 
do
for partial_ratio_male in 0.5
do
for partial_ratio_female in 0.05 0.1 0.2 0.3 0.4 0.5
do
for gender_train_epoch in 1000
do
for seed in 1
do 
main_folder=./MF_results_classifier_contrast_${task_type}_seed_${seed}/change_ratio_and_epoch/partial_ratio_male${partial_ratio_male}/partial_ratio_female_${partial_ratio_female}/gender_train_epoch_${gender_train_epoch}/
mkdir -p ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}
nohup python3 -u ./contrast.py --gpu_id ${gpu_id} --learning_rate $learning_rate --partial_ratio_male $partial_ratio_male --partial_ratio_female $partial_ratio_female \
--gender_train_epoch $gender_train_epoch --weight_decay $weight_decay --fair_reg $fair_reg --saving_path ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}/ \
--result_csv ${main_folder}result.csv --data_path ./datasets/${task_type}/ --orig_unfair_model ./pretrained_model/${task_type}/MF_orig_model --seed ${seed} --task_type ${task_type}> ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}/train.log

done 
done
done
done
done
done
done