top_k=5
gpu_id=1
task_type="Lastfm-360K"
for seed in 1
do 
for learning_rate in 1e-3
do
for weight_decay in 1e-6
do
for fair_reg in 1e-1
do
for partial_ratio_male in  0.3
do
for partial_ratio_female in 0.05
do
for gender_train_epoch in 1000
do
for beta in  0.005
do 

main_folder=./MPR_${task_type}_EXP_seed${seed}/change_ratio_and_epoch/partial_ratio_male${partial_ratio_male}/partial_ratio_female_${partial_ratio_female}/gender_train_epoch_${gender_train_epoch}/
mkdir -p ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_beta_${beta}
nohup python3 -u mpr_safe.py --gpu_id ${gpu_id} --learning_rate $learning_rate --partial_ratio_male $partial_ratio_male --partial_ratio_female $partial_ratio_female \
--gender_train_epoch $gender_train_epoch --weight_decay $weight_decay --fair_reg $fair_reg --beta $beta --task_type ${task_type} \
--saving_path ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_beta_${beta}/ \
--result_csv ${main_folder}result.csv --seed ${seed} --data_path ./datasets/${task_type}/ > ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_beta_${beta}/train.log
done 
done 
done
done
done
done
done
done
done
done 