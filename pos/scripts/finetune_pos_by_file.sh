#! /bin/bash

model_name=bert-base-multilingual-cased
root_ud_dir=/projects/abeb4417/multilingual_analysis/data/pos_ud_data/ud-treebanks-v2.15/

for model_name in bert-base-multilingual-cased xlm-roberta-base; do
    for split in train; do

        langrank_folders=/projects/abeb4417/multilingual_analysis/ma_utils/file_iso_maps/langrank_${split}_folders
        echo ${langrank_folders}

        # Read in all UD folders we want to load, find the right file corresponding to the split of interest, and append to found files
        while IFS= read -r folder; do
            
            curr_dir=${root_ud_dir}${folder}
            echo ${curr_dir}

            for ud_file in "${curr_dir}/"*; do

                if [[ "${ud_file}" == *${split}* && "${ud_file}" == *.conllu ]]; then
                    
                    echo ${ud_file}
                    python run_finetuning_pos.py \
                    --training_config ${CONFIG_DIR}/training_configs/pos_finetune_config.yaml \
                    --log_directory "${PROJECT_DIR}/pos/logs/finetuning_logs/" \
                    --output_directory "${OUTPUT_DIR}" \
                    --experiment_name pos_finetuned_models \
                    --run_name ${model_name} $(basename ${ud_file}) \
                    --model ${model_name} \
                    --train_file "${ud_file}" \
                    --seed 42 \

                fi    
            done

        done < ${langrank_folders}


    done
done