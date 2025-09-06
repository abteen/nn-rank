#!/bin/bash

# Source bible target bible
for model in xlm-roberta-base bert-base-multilingual-cased; do
    for target_split in dev_all; do
            for source_split in train_large train_medium train_all; do
                    model=${model} target_split=${target_split} source_split=${source_split} \
                    sbatch -J bible-ner-${model:0:5}-nn-${source_split} -o jobs/slurm_outputs/bible_results/ner/source_rahimi_target_bible/${model}/${source_split}_${target_split}/S-%x-%j.out \
                    --array 0-9 -N1 jobs/bible_ner_embedding_nn_efficient.slurm
            done
    done
done

