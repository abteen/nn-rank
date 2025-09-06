#!/bin/bash


for split in test dev; do
    for model in bert-base-multilingual-cased xlm-roberta-base; do
        log_dir_to_read=/projects/abeb4417/multilingual_analysis/ner/logs/evaluation_logs/ner_evaluation/${model}/${split}/
        save_dir=/projects/abeb4417/multilingual_analysis/ner/outputs/ner_evaluation_outputs/${model}/${split}
        save_fname=results_${model}_${split}.csv

        python process_ner_logs.py ${log_dir_to_read} --save_dir ${save_dir} --save_fname ${save_fname} --eval_split ${split} --source_split train


    done
done