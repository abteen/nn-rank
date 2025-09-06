# Model-Based Ranking of Source Languages for Zero-Shot Cross-Lingual Transfer

This repository contains the code for NN-Rank. 

## Python Package

To install the package:
```
    cd nnrank/
    pip install -e .
```

The `nnrank/example_scripts` folder shows different ways to use the package.

## Reproducing Paper Results

The `nn_calculations/` folder contains the experiment code used to produce the results shown in the paper, with `scripts/` containing bash scripts used to run each Python script. In general, the process is to (1) extract hidden representations, (2) calculate the nearest neighbors, and (3) process this output and produce a ranking. The [MWE](#nn-rank-minimum-working-example) contains more details on how to use the original experimental code. 

### Minimum Working Example

1. Setup environment variables using `startup.sh`:
    - PROJECT_DIR: location of repository root
    - OUTPUT_DIR: directory where hidden representations are saved (should not have storage restrictions)

2. Extract hidden representations using `scripts/extract_hidden_reps.sh`
    - This script runs `extract_hidden_reps_nn_efficient.py`. The input is a model and list of files to extract representations from. The output is one file per line of the input. 
    - `run_name_extractor` is used to set the function which creates the output file name given the input file

3. Calculate nearest neighbors using `scripts/embedding_nn.sh`
    - Here the input is a list of source directories and target directories. If using your own data, a function must be defined and added to `iso2dir` which, for each directory, returns an ISO and dataset identifier. The ISO is used to make sure that the target language does not appear in the source pool, while the identifier should be a unique value. The method is included for bible files, please see `nn_calculations/` for NER and UD implementations.
    - This script does not require GPU resources. Therefore, you can use the `num_shards` and `shard_idx` to distribute the calculation over different compute nodes. 
    - A description of the output files can be found in the python script. 

4. Calculate ranking using `scripts/process_nn_outputs.sh`
    - `method` defines how the neighbors contribute to the tally.
        -  `simple_global_neighbor_distribution`: each neighbor contributes 1 to the tally
        - `sgnd-verbose`: equivalent to the above, but prints out the neighbors. helpful for sanity checks/analysis; currently uses mBERT tokenizer to show surface forms
        - `simple_global_neighbor_distribution_weighted`: (does not appear in paper) same as the first approach, but inversely weights neighbors by distance. initial tests did not show any improvement over the baseline version.
    - Produces a dictionary which maps each source dataset to its tally. To generate the ranking, sort the keys in descending order.
    - Important note: if a source datset did not appear in the nearest neighbors for *any* target token, then it will not appear in the dictionary/will not be ranked (e.g., Coptic was included in the source pool, but does not appear in the tally).




## Citation

*To appear: EMNLP 2025*
