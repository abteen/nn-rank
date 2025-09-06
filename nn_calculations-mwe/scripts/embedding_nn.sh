#! /bin/bash

num_shards=1
shard_idx=0
layer=8
experiment_name=mwe
model=bert-base-multilingual-cased

source_pool=${OUTPUT_DIR}/${experiment_name}/${model}/source_pool
target_pool=${OUTPUT_DIR}/${experiment_name}/${model}/target_datasets

declare -a source_directories
declare -a target_directories

for d in ${source_pool}/*; do
    if [[ -d ${d} ]]; then
        source_directories+=("${d} ")
    fi 

done


for d in ${target_pool}/*; do
    if [[ -d ${d} ]]; then
        target_directories+=("${d} ")
    fi 

done


echo "Found ${#source_directories[@]} valid source directories"
echo "Found ${#target_directories[@]} valid target directories"


python embedding_nn_efficient.py \
    --target_directories ${target_directories[@]} \
    --source_directories ${source_directories[@]} \
    --source_directory_type bible \
    --target_directory_type bible \
    --root_output_dir ${OUTPUT_DIR} \
    --experiment_name mwe_nn \
    --run_name ${model} layer_${layer} ${source_split} ${target_split} \
    --remove_target_from_source_langs \
    --load_target_languages_independently \
    --limiter 1000 \
    --k 25 \
    --layer ${layer} \
    --num_shards ${num_shards} \
    --shard_idx ${shard_idx} \
    --check_already_trained