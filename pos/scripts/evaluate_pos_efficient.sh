#! /bin/bash


# split=dev
# base_model=xlm-roberta-base
root_ud_dir=/projects/abeb4417/multilingual_analysis/data/pos_ud_data/ud-treebanks-v2.15

for split in test dev; do
    for base_model in bert-base-multilingual-cased xlm-roberta-base; do


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

        echo ${found_eval_files}

        model_load_dir=/scratch/alpine/abeb4417/multilingual_analysis/pos_finetuned_models/${base_model}/

        for model in "${model_load_dir}"/*; do

            if [[ -d "${model}" ]]; then

                for load_dir in "${model}"/*; do
                
                        if [[ "${load_dir}" == *"final_model"* ]]; then

                                python run_evaluation_pos_efficient.py \
                                    --experiment_name pos_evaluation_rerun \
                                    --run_name ${base_model} ${split} \
                                    --log_directory="${PROJECT_DIR}/pos/logs/evaluation_logs/" \
                                    --output_directory="${OUTPUT_DIR}" \
                                    --model_to_evaluate=${load_dir} \
                                    --base_model=${base_model} \
                                    --eval_files ${found_eval_files[@]} 

                            echo ${load_dir}

                        fi 

                done

            fi
        done

    done
done