#! /bin/bash

root_ner_dir=/projects/abeb4417/multilingual_analysis/data/ner

split=${1}
base_model=${2}


        langrank_folders=/projects/abeb4417/multilingual_analysis/ma_utils/ner_file_iso_maps/langrank_${split}_all_folders
        # echo ${langrank_folders}
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

        echo ${found_eval_files}

        model_load_dir=/scratch/alpine/abeb4417/multilingual_analysis/ner_finetuned_models/${base_model}

        for model in "${model_load_dir}"/*; do
            echo ${model}
            if [[ -d "${model}" ]]; then

                for load_dir in "${model}"/*; do
                
                        if [[ "${load_dir}" == *"final_model"* ]]; then

                                python run_evaluation_ner_efficient.py \
                                    --experiment_name ner_evaluation_eff \
                                    --run_name ${base_model} ${split} \
                                    --log_directory="${PROJECT_DIR}/ner/logs/evaluation_logs/" \
                                    --output_directory="${OUTPUT_DIR}" \
                                    --model_to_evaluate=${load_dir} \
                                    --base_model=${base_model} \
                                    --eval_files ${found_eval_files[@]}

                            echo ${load_dir}

                        fi 

                done

            fi
        done
