#! /bin/bash

model_name=bert-base-multilingual-cased
root_ner_dir=/projects/abeb4417/multilingual_analysis/data/ner/
layer=8
split=dev

for model_name in xlm-roberta-base bert-base-multilingual-cased; do
    for split in dev test train; do

        langrank_folders=/projects/abeb4417/multilingual_analysis/ma_utils/el_file_iso_maps/langrank_${split}_folders
        echo ${langrank_folders}

        found_files=()

        # Read in all UD folders we want to load, find the right file corresponding to the split of interest, and append to found files
        while IFS= read -r folder; do
            
            curr_dir=${root_ner_dir}${folder}
            echo ${curr_dir}

            for ner_file in "${curr_dir}/"*; do

                if [[ "${ner_file}" == *${split}* && "${ner_file}" == *.txt ]]; then
                    
                    echo ${ner_file}
                    found_files+="${ner_file} "

                fi    
            done

        done < ${langrank_folders}

        echo ${found_files}


        python extract_hidden_reps_nn_efficient.py ${model_name} ${found_files} \
                    --root_output_dir /scratch/alpine/abeb4417/multilingual_analysis/ \
                    --run_name_extractor ner \
                    --experiment_name ner_hidden_reps \
                    --run_name ${model_name} ${split} \
                    --limiter 1000 \
                    --layer ${layer}

    done
done