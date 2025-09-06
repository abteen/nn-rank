#! /bin/bash

layer=8
model_name=bert-base-multilingual-cased

# Extract Hidden Reps for Source Datasets #

source_pool=${MWE_DIR}/mwe_data/source_pool
echo source pool: ${source_pool}

declare -a found_source_files

# Read in all Bible folders we want to load and append to found files
for source_file in ${source_pool}/*.txt; do

    found_source_files+=("${source_file} ")

done

echo ${found_source_files[@]}
echo found ${#found_source_files[@]} source files


python extract_hidden_reps_nn_efficient.py ${model_name} ${found_source_files[@]} \
            --root_output_dir ${OUTPUT_DIR} \
            --experiment_name mwe \
            --run_name ${model_name} source_pool \
            --run_name_extractor bible \
            --limiter 1000 \
            --layer ${layer}

echo done with source pool.

# Extract Hidden Reps for Target Dataset #

target_datasets=${MWE_DIR}/mwe_data/target_datasets
echo target dataset: ${target_dataset}

declare -a found_target_files

# Read in all Bible folders we want to load and append to found files
for target_file in ${target_datasets}/*.txt; do

    found_target_files+=("${target_file} ")

done

echo ${found_target_files[@]}
echo found ${#found_target_files[@]} source files


python extract_hidden_reps_nn_efficient.py ${model_name} ${found_target_files[@]} \
            --root_output_dir ${OUTPUT_DIR} \
            --experiment_name mwe \
            --run_name ${model_name} target_datasets \
            --run_name_extractor bible \
            --limiter 1000 \
            --layer ${layer}

echo done with target pool.

