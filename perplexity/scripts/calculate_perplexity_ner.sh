#!/bin/bash

root_ud_dir='/projects/abeb4417/multilingual_analysis/data/ner/'


for split_to_evaluate in train_all dev_all test_all; do
    
    folders_to_read="/projects/abeb4417/multilingual_analysis/ma_utils/ner_file_iso_maps/langrank_${split_to_evaluate}_folders"
    
    for model in xlm-roberta-base bert-base-multilingual-cased; do
        
        while IFS= read -r folder; do
            echo ${folder}

            IFS="_" read -r split_name split_size <<< ${split_to_evaluate} #split dev_all into dev, all

            curr_dir=${root_ud_dir}${folder}

                    for ud_file in "${curr_dir}/"*; do

                        if [[ "${ud_file}" == *${split_name}* && "${ud_file}" == *.txt ]]; then
                            
                            python calculate_perplexity.py \
                                ${ud_file} \
                                --model_name ${model} \
                                --experiment_name ner_pppl \
                                --run_name ${split_to_evaluate} ${model} $(basename ${folder}) \
                                --subsample 500

                        fi    
                    done


                
        done < ${folders_to_read}
    done
done

