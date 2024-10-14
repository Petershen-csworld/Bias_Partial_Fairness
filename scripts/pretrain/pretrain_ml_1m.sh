top_k=5
gpu_id=0
task_type="ml-1m"
for learning_rate in 1e-3
do
for weight_decay in 1e-5
do
for fair_reg in 0
do
for partial_ratio in 1.0 # partial_ratio doesn't matter since fair_reg is 0
do

main_folder=./MF_results_${task_type}/gender_shuffle/partial_iid/partial_ratio_${partial_ratio}/
mkdir -p ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}
nohup python3 -u MF_explicit_fairness_fast_eval_partial_shuffle.py --gpu_id ${gpu_id} --learning_rate $learning_rate --partial_ratio_male $partial_ratio --partial_ratio_female $partial_ratio \
--weight_decay $weight_decay --fair_reg $fair_reg --data_path ./datasets/${task_type}/ --saving_path ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}/ \
--result_csv ${main_folder}result.csv --task_type ${task_type}> ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}/train.log

done
done
done
done