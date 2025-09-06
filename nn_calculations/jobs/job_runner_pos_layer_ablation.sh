#!/bin/bash
for model in xlm-roberta-base bert-base-multilingual-cased; do
    for target_split in dev_all; do
            for source_split in train_large train_medium train_all; do
                    model=${model} target_split=${target_split} source_split=${source_split} \
                    sbatch -J pos-${model:0:5}-nn-${target_split:3:5} -o jobs/slurm_outputs/layer-ablations/pos/${model}/${source_split}_${target_split}/S-%x-%j.out \
                    --array 0-9 -N1 jobs/layer_pos_embedding_nn_efficient.slurm
            done
    done
done
