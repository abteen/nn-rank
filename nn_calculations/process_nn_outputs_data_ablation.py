import argparse
import numpy as np
import os
import orjson, json
import collections
from tqdm import tqdm

def get_token_idx_from_line(line_idx, indices):
    return np.where(indices == line_idx)

def get_token_input_ids_from_token_ids(token_indices, input_ids):
    return input_ids[token_indices]

def get_token_input_ids_from_line(line_idx, indices, input_ids):
    return input_ids[np.where(indices == line_idx)]

def get_all_line_idx(indices):
    return np.unique(indices)

def get_first_available_line(indices):
    return indices[0]

def load_data(load_dir):
    data = {}
    
    for found_file in os.listdir(load_dir):

        full_file_path = os.path.join(load_dir, found_file)

        if '.npy' in found_file:

            v_name = found_file[:-4]
            v = np.load(full_file_path)

            data[v_name] = v

        elif '.jsonl' in found_file:
            v_name = found_file[:-6]
            with open(full_file_path, 'r') as f:
                for line in f:
                    v = orjson.loads(line.strip())
                    break #  Assume only 1 line of information, must edit if this changes
            data[v_name] = v
            
    return data

def simple_global_neighbor_distribution_subsampled(loaded_data, k, seed, subsample_size):
    """
        Baseline approach. 
        For each target token, lookup the nearest neighbors. 
        For each neighbor, lookup it's associated train dataset
        Update the distribution of train sets

    """

    lang_distribution = collections.Counter()

    logging_warn_k = False
    num_neighbors = 0


    print(f'Given {len(loaded_data["target_indices"])} target indices to start with.')
    print(f'Setting random seed to {seed}')

    rng = np.random.default_rng(seed=seed)

    # Recall that all target_* arrays, neighbors, and distances are indexed by the seen target tokens
    # Therefore, we sample from the *length* of target_indices (which is equal to the length of the neighbors, so that would be a valid alternate)
    # Samples from the index --> correspond to sampling target tokens, which we then lookup only the neighbors of each sampled target token


    subsampled_target = rng.choice(len(loaded_data['target_indices']), size=subsample_size, replace=False)

    print(f'Length of subsampled target: {subsampled_target.shape}, with values:')
    print(subsampled_target)

    subsampled_input_ids = list(loaded_data['target_input_ids'][subsampled_target])
    neighbor_input_ids = {}
    neighbor_langs = {}

    for idx in subsampled_target:

        neighbors = loaded_data['neighbors_indices'][idx] if k < 0 else loaded_data['neighbors_indices'][idx][:k]
        neighbor_input_ids[str(idx)] = list(loaded_data['source_input_ids'][neighbors])
        neighbor_langs[str(idx)] = list(loaded_data['source_langs'][neighbors])
        

        if k > 0 and loaded_data['neighbors_indices'][idx].shape[-1] < k:
            logging_warn_k = True
            num_neighbors = loaded_data['neighbors_indices'][idx].shape[-1]
        lang_distribution.update(loaded_data['source_langs'][neighbors])

    if logging_warn_k == True:
        print('#'*50 + f'\n Given k value of {k}, which is greater than number of available neighbors {num_neighbors}. This means that the ranking will be using a k value equivalent to {num_neighbors}.\n' + '#'*50)
    return lang_distribution, subsampled_input_ids, neighbor_input_ids, neighbor_langs, list(subsampled_target)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_dir')
    parser.add_argument('--target_lang')
    parser.add_argument('--method')
    parser.add_argument('--k', type=int, default=-1, help='Set the max number of neighbors to consider.')
    parser.add_argument('--join_target_lang_and_load_dir', action='store_const', const=True, default=False)
    parser.add_argument('--save_dir')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsample_size', type=int)

    args = parser.parse_args()

    method_factory = {
        'simple_global_neighbor_distribution_subsampled' : simple_global_neighbor_distribution_subsampled,
        
    }

    if args.method not in method_factory.keys():
        print(f'Unsupported method. Pick from one of: {",".join(method_factory.keys())}')
        quit()

    method = method_factory[args.method]

    load_dir = args.load_dir if not args.join_target_lang_and_load_dir else os.path.join(args.load_dir, args.target_lang)

    data = load_data(load_dir)

    predicted_source_rankings, subsampled_input_ids, neighbor_input_ids, neighbor_langs, subsample_idx = method(data, k=args.k, seed=args.seed, subsample_size=args.subsample_size)
    original_dtypes = {}


    for k in list(predicted_source_rankings.keys()):
        psr = predicted_source_rankings.pop(k)

        
        if isinstance(psr, np.floating):
            original_dtypes[str(k)] = psr.dtype.descr

        elif isinstance(psr, int):
            original_dtypes[str(k)] = 'int'

        predicted_source_rankings[str(k)] = str(psr)

    metadata = vars(args)

    final_save_data = {'results' : predicted_source_rankings, 'dtypes' : original_dtypes, 'subsampled_target_input_ids' : subsampled_input_ids, 'neighbor_input_ids' : neighbor_input_ids, 'neighbor_langs' : neighbor_langs, 'subsample_target_idx' : subsample_idx} | metadata

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, args.target_lang + '.jsonl'), 'w') as f:
        f.write(json.dumps(final_save_data, default=str))



