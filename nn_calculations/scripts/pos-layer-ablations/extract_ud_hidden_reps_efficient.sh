#! /bin/bash


root_ud_dir=/projects/abeb4417/multilingual_analysis/data/pos_ud_data/ud-treebanks-v2.15/
layer=0

for model_name in xlm-roberta-base bert-base-multilingual-cased ; do
    for split in train dev; do

        langrank_folders=/projects/abeb4417/multilingual_analysis/ma_utils/pos_file_iso_maps/langrank_${split}_all_folders
        echo ${langrank_folders}

        found_files=()

        # Read in all UD folders we want to load, find the right file corresponding to the split of interest, and append to found files
        while IFS= read -r folder; do
            
            curr_dir=${root_ud_dir}${folder}
            echo ${curr_dir}

            for ud_file in "${curr_dir}/"*; do

                if [[ "${ud_file}" == *${split}* && "${ud_file}" == *.txt ]]; then
                    
                    echo ${ud_file}
                    found_files+="${ud_file} "

                fi    
            done

        done < ${langrank_folders}


        python extract_hidden_reps_nn_efficient.py ${model_name} ${found_files} \
                    --root_output_dir /scratch/alpine/abeb4417/multilingual_analysis/ \
                    --experiment_name layer_ablations/ud_hidden_reps_0 \
                    --run_name ${model_name} ${split} \
                    --run_name_extractor ud \
                    --limiter 1000 \
                    --layer ${layer}

    done
done