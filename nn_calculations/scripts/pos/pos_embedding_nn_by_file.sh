#! /bin/bash

# Experiment 1: Compare the representations of the dev subwords to all train subwords (selecting a source language given a target).
# Restricted to eval and source languages supported by pretrained langrank
num_shards=${1}
shard_idx=${2}
model=${3}
source_split=${4}
target_split=${5}

# target_directories="/scratch/alpine/abeb4417/multilingual_analysis/ud_hidden_reps/${model}/${target_split}/"*${target_split}*
# source_directories="/scratch/alpine/abeb4417/multilingual_analysis/ud_hidden_reps/${model}/${source_split}/"*${source_split}*

target_file_list_to_read="/projects/abeb4417/multilingual_analysis/ma_utils/pos_file_iso_maps/langrank_${target_split}_fnames"
source_file_list_to_read="/projects/abeb4417/multilingual_analysis/ma_utils/pos_file_iso_maps/langrank_${source_split}_fnames"

declare -a target_directories
declare -a source_directories

IFS="_" read -r target_split_name target_split_size <<< ${target_split}
IFS="_" read -r source_split_name source_split_size <<< ${source_split}


while IFS= read -r fname; do
    dir_to_read="/scratch/alpine/abeb4417/multilingual_analysis/ud_hidden_reps/${model}/${target_split_name}/${fname}"

    if [[ ! -d ${dir_to_read} ]]; then
        echo "ERROR: DIR NOT FOUND: ${dir_to_read}"
    else
        target_directories+=("${dir_to_read} ")
    fi

done < ${target_file_list_to_read}

while IFS= read -r fname; do
    dir_to_read="/scratch/alpine/abeb4417/multilingual_analysis/ud_hidden_reps/${model}/${source_split_name}/${fname}"

    if [[ ! -d ${dir_to_read} ]]; then
        echo "ERROR: DIR NOT FOUND: ${dir_to_read}"
    else
        source_directories+=("${dir_to_read} ")
    fi

done < ${source_file_list_to_read}

echo "Found ${#target_directories[@]} valid target directories"
echo "Found ${#source_directories[@]} valid source directories"


layer=8

echo ${source_directories}

python embedding_nn_efficient.py \
    --target_directories ${target_directories[@]} \
    --source_directories ${source_directories[@]} \
    --root_output_dir /scratch/alpine/abeb4417/multilingual_analysis/ud_nn_outputs/ \
    --experiment_name ud_pos_nn_sized \
    --run_name ${model} layer_${layer} ${source_split} ${target_split} \
    --remove_target_from_source_langs \
    --load_target_languages_independently \
    --limiter 1000 \
    --k 25 \
    --layer ${layer} \
    --num_shards ${num_shards} \
    --shard_idx ${shard_idx} \
    --check_already_trained