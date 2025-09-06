#!/bin/bash

split=dev
base_model=xlm-roberta-base
root_ud_dir=/projects/abeb4417/multilingual_analysis/data/pos_ud_data/ud-treebanks-v2.15

langrank_folders=/projects/abeb4417/multilingual_analysis/ma_utils/pos_file_iso_maps/langrank_${split}_folders
echo ${langrank_folders}
found_eval_files=()

# Read in all UD folders we want to load, find the right file corresponding to the split of interest, and append to found files
while IFS= read -r folder; do
    
    curr_dir=${root_ud_dir}/${folder}
    # echo ${curr_dir}

    for ud_file in "${curr_dir}/"*; do

        if [[ "${ud_file}" == *${split}* && "${ud_file}" == *.conllu ]]; then
            
            # echo ${ud_file}
            found_eval_files+="${ud_file} "

        fi    
    done

done < ${langrank_folders}

python pos_datasets.py ${found_eval_files[@]} 