#! /bin/bash

root_ner_dir=/projects/abeb4417/multilingual_analysis/data/ner/

for model_name in xlm-roberta-base bert-base-multilingual-cased; do
    for split in train; do

        langrank_folders=/projects/abeb4417/multilingual_analysis/ma_utils/ner_file_iso_maps/langrank_${split}_all_folders
        echo ${langrank_folders}

        # Read in all NER folders we want to load, find the right file corresponding to the split of interest
        while IFS= read -r folder; do
            
            curr_dir=${root_ner_dir}${folder}
            echo current folder name: ${folder}
            echo current directory path: ${curr_dir}


            for ner_file in "${curr_dir}/"*; do

                if [[ "${ner_file}" == *${split}* && ! "${ner_file}" == *.txt ]]; then
                    echo selected file for training: ${ner_file}
                    
                    python run_finetuning_ner.py \
                    --training_config ${CONFIG_DIR}/training_configs/ner_finetune_config.yaml \
                    --log_directory "${PROJECT_DIR}/ner/logs/finetuning_logs/" \
                    --output_directory "${OUTPUT_DIR}" \
                    --experiment_name ner_finetuned_models \
                    --run_name ${model_name} ${folder} \
                    --train_iso ${folder} \
                    --model ${model_name} \
                    --train_file "${ner_file}" \
                    --seed 42 \

                fi    
            done

        done < ${langrank_folders}


    done
done