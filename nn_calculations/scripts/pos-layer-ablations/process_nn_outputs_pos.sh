#! /bin/bash

experiment=ud_pos_nn_sized_0

for model in bert-base-multilingual-cased xlm-roberta-base; do
    for method in simple_global_neighbor_distribution; do
        for layer in layer_0; do
            for train_split in train_all train_large train_medium; do
                for eval_split in dev_all; do 
                    for k in 5; do

                        echo "currently working on ${train_split} ${eval_split} ${k}"

                        root_load_dir=/scratch/alpine/abeb4417/multilingual_analysis/layer_ablations/${experiment}/${model}/${layer}/${train_split}/${eval_split}/
                        i=0
                        n=$(find "${root_load_dir}" -maxdepth 1 | wc -l)
                        ((n=n-1))

                        echo "Total: ${n}"
                        for target_lang_dir in "${root_load_dir}"/*; do 
                            
                            target_lang=$(basename $target_lang_dir)
                            echo ${target_lang}

                            
                            python process_nn_outputs.py \
                                --load_dir ${target_lang_dir} \
                                --target_lang ${target_lang} \
                                --method ${method} \
                                --k ${k} \
                                --save_dir layer_ablation_outputs/pos/${experiment}/${model}/${train_split}/${eval_split}/${method}/${layer}/k_${k}/
                            
                            ((i+=1))
                            echo "${i}/${n}"

                            
                    
                        done
                    done
                done
            done 
        done
    done
done