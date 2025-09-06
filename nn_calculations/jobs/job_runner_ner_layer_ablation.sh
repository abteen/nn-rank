#!/bin/bash
for model in xlm-roberta-base bert-base-multilingual-cased; do
    for target_split in dev_all; do # Since medium and large target langs are contained in all, we only need to run it on *_all (for eval splits)
            for source_split in train_large train_medium train_all; do # we actually have to consider the different soruce splits here, because it changes the pool
                    model=${model} target_split=${target_split} source_split=${source_split} \
                    sbatch -J ner-${model:0:5}-nn-${target_split:4:6} -o jobs/slurm_outputs/layer-ablations/ner/${model}/${source_split}_${target_split}/S-%x-%j.out \
                    --array 0-9 -N1 jobs/layer_ner_embedding_nn_efficient.slurm

            done
    done
done
