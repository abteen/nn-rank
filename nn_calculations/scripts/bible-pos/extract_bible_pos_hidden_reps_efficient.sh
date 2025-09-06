#! /bin/bash


root_bible_dir='/projects/abeb4417/old/lrl/data/bibles_raw'
layer=8

for model_name in xlm-roberta-base bert-base-multilingual-cased; do
    for split in dev train; do

        bible_folders=/projects/abeb4417/multilingual_analysis/ma_utils/bible_file_iso_maps/pos/langrank_bible_${split}_all_columns
        echo ${bible_folders}

        found_files=()

        # Read in all Bible folders we want to load, find the right file corresponding to the split of interest, and append to found files
        while IFS= read -r bible_row; do
            bible_file_to_load=$(echo "${bible_row}" | awk -F"\t" '{print $5}')

            for bible_file in "${root_bible_dir}/"*; do

                if [[ "${bible_file}" == *${bible_file_to_load} ]]; then
                    found_files+="${bible_file} "

                fi    
            done

        done < ${bible_folders}

        echo ${found_files}

        python extract_hidden_reps_nn_efficient.py ${model_name} ${found_files} \
                    --root_output_dir /scratch/alpine/abeb4417/multilingual_analysis/ \
                    --experiment_name bible_pos_hidden_reps \
                    --run_name ${model_name} ${split} \
                    --run_name_extractor bible \
                    --limiter 1000 \
                    --layer ${layer}

    done
done