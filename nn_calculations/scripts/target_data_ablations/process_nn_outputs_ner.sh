#! /bin/bash

experiment=ner_nn_sized

for model in bert-base-multilingual-cased xlm-roberta-base; do
    for method in simple_global_neighbor_distribution_subsampled; do
        for layer in layer_8; do
            for train_split in train_all train_large train_medium; do
                for eval_split in dev_all; do 
                    for seed in 42 925 963; do
                        for subsample_size in 10 25 50 75 100 150 250 500 1000 2000; do
                            for k in 5; do

                                echo "currently working on ${train_split} ${eval_split} ${k}"

                                root_load_dir=/scratch/alpine/abeb4417/multilingual_analysis/ner_nn_outputs/${experiment}/${model}/${layer}/${train_split}/${eval_split}/
                                i=0
                                n=$(find "${root_load_dir}" -maxdepth 1 | wc -l)
                                ((n=n-1))

                                echo "Total: ${n}"
                                for target_lang_dir in "${root_load_dir}"/*; do 
                                    
                                    target_lang=$(basename $target_lang_dir)
                                    echo ${target_lang}

                                    
                                    python process_nn_outputs_data_ablation.py \
                                        --load_dir ${target_lang_dir} \
                                        --target_lang ${target_lang} \
                                        --method ${method} \
                                        --k ${k} \
                                        --seed ${seed} \
                                        --subsample_size ${subsample_size} \
                                        --save_dir data_ablation_outputs_with_tokens/ner/${experiment}/seed_${seed}/${model}/${train_split}/${eval_split}/${method}/${layer}/k_${k}/subsample_${subsample_size}
                                    
                                    ((i+=1))
                                    echo "${i}/${n}"
                                   
                            
                                done
                            done
                        done
                    done
                done
            done 
        done
    done
done