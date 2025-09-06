#! /bin/bash

experiment=mwe_nn
model=bert-base-multilingual-cased
method=simple_global_neighbor_distribution
layer=layer_8
k=5

echo "currently working on ${experiment} ${model} ${k}"

root_load_dir=${OUTPUT_DIR}/${experiment}/${model}/${layer}/
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
        --save_dir ranking/${experiment}/${model}/${method}/${layer}/k_${k}/
    
    ((i+=1))
    echo "${i}/${n}"
done
