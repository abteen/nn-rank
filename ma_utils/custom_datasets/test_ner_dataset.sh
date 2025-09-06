#!/bin/bash

split=dev
base_model=bert-base-multilingual-cased
root_ner_dir=/projects/abeb4417/multilingual_analysis/data/ner

langrank_folders=/projects/abeb4417/multilingual_analysis/ma_utils/el_file_iso_maps/langrank_${split}_folders
echo ${langrank_folders}
found_eval_files=()

# Read in all UD folders we want to load, find the right file corresponding to the split of interest, and append to found files
while IFS= read -r folder; do
    
    curr_dir=${root_ner_dir}/${folder}
    # echo ${curr_dir}

    for ner_file in "${curr_dir}/"*; do

        if [[ "${ner_file}" == *${split}* && ! "${ner_file}" == *.txt ]]; then
            
            # echo ${ner_file}
            found_eval_files+="${ner_file} "

        fi    
    done

done < ${langrank_folders}

# echo ${found_eval_files[@]}
python ner_datasets.py ${found_eval_files[@]} 