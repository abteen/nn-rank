#!/bin/bash
for model in xlm-roberta-base bert-base-multilingual-cased; do
    for target_split in test_large test_medium test_all dev_large dev_medium dev_all; do
            for source_split in train_large train_medium train_all; do
                    model=${model} target_split=${target_split} source_split=${source_split} \
                    sbatch -J pos-${model}-nn-${target_split} -o jobs/slurm_outputs/main_results-sized/pos/${model}/${source_split}_${target_split}/S-%x-%j.out \
                    --array 0-9 -N1 jobs/pos_embedding_nn_efficient.slurm
            done
    done
done
