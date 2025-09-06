#!/bin/bash


for split in dev test; do
    for model in xlm-roberta-base bert-base-multilingual-cased; do
        log_dir_to_read=/projects/abeb4417/multilingual_analysis/pos/logs/evaluation_logs/pos_evaluation_rerun/${model}/${split}/
        save_dir=/projects/abeb4417/multilingual_analysis/pos/outputs/pos_evaluation_outputs/${model}/${split}
        save_fname=results_${model}_${split}.csv

        python process_pos_logs.py ${log_dir_to_read} --save_dir ${save_dir} --save_fname ${save_fname}


    done
done