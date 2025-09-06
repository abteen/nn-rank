#!/bin/bash

#SBATCH --ntasks=32
#SBATCH --mem=128g
#SBATCH --time=10:00:00
#SBATCH --qos=preemptable
#SBATCH -J default
#SBATCH -o jobs/slurm_outputs/default/S-%x-%j.out


cd /projects/abeb4417/multilingual_analysis/
source startup.sh;

cd nn_calculations/

num_shards=10

# num_shards=${1}
# shard_idx=${2}
# model=${3}
# source_split=${4}
# target_split=${5}

./scripts/bible-ner/source_rahimi_target_bible_ner_embedding_nn_by_file.sh ${num_shards} $SLURM_ARRAY_TASK_ID ${model} ${source_split} ${target_split}


