import argparse
import numpy as np
import os
import orjson
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

def simple_global_neighbors_by_line_verbose(loaded_data):
    """ 
        Identical result to simple_global_neighbors, however iterates line-by-line, then token-by-token.
        Keeping as guide for future methods. 
    """
    all_line_indices = np.unique(loaded_data['target_indices'])

    lang_distribution = collections.Counter()

    for line_idx in all_line_indices:

        target_token_ids = get_token_idx_from_line(line_idx, loaded_data['target_indices'])[0] # The tokens in line 0 are mapped to the indices stored in target_token_ids
        # print(target_token_ids)
        # use these indices to look up their input ids:

        target_input_ids = get_token_input_ids_from_token_ids(target_token_ids, loaded_data['target_input_ids'])
        # print(target_input_ids)

        # Sanity check: convert these ids to the tokens:
        # print(tokenizer.convert_ids_to_tokens(target_input_ids))

        # For each token, lets get their nearest neighbors

        for i in range(target_token_ids.shape[0]):
            # print('#####'*50)

            target_token_id = target_token_ids[i]
            target_token_input_id = target_input_ids[i]

            # print(target_token_id)
            # print(target_token_input_id)
            # print(tokenizer.convert_ids_to_tokens([target_token_input_id]))

            neighbors = loaded_data['neighbors_indices'][target_token_id]
            # print(neighbors)

            neighbor_input_ids = loaded_data['source_input_ids'][neighbors]
            # print(neighbor_input_ids)
            # print(tokenizer.convert_ids_to_tokens(neighbor_input_ids))

            # print(loaded_data['source_langs_input_id'][neighbors])
            lang_distribution.update(loaded_data['source_langs'][neighbors])


    return lang_distribution

def simple_global_neighbor_distribution(loaded_data, k):
    """
        Baseline approach. 
        For each target token, lookup the nearest neighbors. 
        For each neighbor, lookup it's associated train dataset
        Update the distribution of train sets

    """

    lang_distribution = collections.Counter()

    logging_warn_k = False
    num_neighbors = 0

    for idx in range(len(loaded_data['target_indices'])):

        neighbors = loaded_data['neighbors_indices'][idx] if k < 0 else loaded_data['neighbors_indices'][idx][:k]
        

        if k > 0 and loaded_data['neighbors_indices'][idx].shape[-1] < k:
            logging_warn_k = True
            num_neighbors = loaded_data['neighbors_indices'][idx].shape[-1]
        lang_distribution.update(loaded_data['source_langs'][neighbors])

    if logging_warn_k == True:
        print('#'*50 + f'\n Given k value of {k}, which is greater than number of available neighbors {num_neighbors}. This means that the ranking will be using a k value equivalent to {num_neighbors}.\n' + '#'*50)
    return lang_distribution

def simple_global_neighbor_distribution_weighted(loaded_data, k):
    """
        Baseline approach weighted by inverse distance. 
        For each target token, lookup the nearest neighbors. 
        For each neighbor, lookup it's associated train dataset
        Update the distribution of train sets, weighted by inverse distances

    """

    lang_distribution = collections.Counter()

    logging_warn_k = False
    num_neighbors = 0

    for idx in range(len(loaded_data['target_indices'])):

        neighbors = loaded_data['neighbors_indices'][idx] if k < 0 else loaded_data['neighbors_indices'][idx][:k]
        neighbor_langs = loaded_data['source_langs'][neighbors] # no need to trim since lookup indices are trimmed above

        distances = neighbors = loaded_data['distances'][idx] if k < 0 else loaded_data['distances'][idx][:k]

        
        with np.errstate(divide='ignore'): #https://stackoverflow.com/a/69814700
            inverse_distances = np.nan_to_num(1 / distances)

        for nl,invd in zip(neighbor_langs, inverse_distances):

            if k > 0 and loaded_data['neighbors_indices'][idx].shape[-1] < k:
                logging_warn_k = True
                num_neighbors = loaded_data['neighbors_indices'][idx].shape[-1]
            lang_distribution.update({nl:invd})

    if logging_warn_k == True:
        print('#'*50 + f'\nGiven k value of {k}, which is greater than number of available neighbors {num_neighbors}. This means that the ranking will be using a k value equivalent to {num_neighbors}.\n' + '#'*50)

    return lang_distribution

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_dir')
    parser.add_argument('--target_lang')
    parser.add_argument('--method')
    parser.add_argument('--k', type=int, default=-1, help='Set the max number of neighbors to consider.')
    parser.add_argument('--join_target_lang_and_load_dir', action='store_const', const=True, default=False)
    parser.add_argument('--save_dir')

    args = parser.parse_args()

    method_factory = {
        'simple_global_neighbor_distribution' : simple_global_neighbor_distribution,
        'simple_global_neighbor_distribution_weighted' : simple_global_neighbor_distribution_weighted
    }

    if args.method not in method_factory.keys():
        print(f'Unsupported method. Pick from one of: {",".join(method_factory.keys())}')
        quit()

    method = method_factory[args.method]

    load_dir = args.load_dir if not args.join_target_lang_and_load_dir else os.path.join(args.load_dir, args.target_lang)

    data = load_data(load_dir)

    predicted_source_rankings = method(data, k=args.k)
    original_dtypes = {}

    print(predicted_source_rankings)

    for k in list(predicted_source_rankings.keys()):
        psr = predicted_source_rankings.pop(k)

        
        if isinstance(psr, np.floating):
            original_dtypes[str(k)] = psr.dtype.descr

        elif isinstance(psr, int):
            original_dtypes[str(k)] = 'int'

        predicted_source_rankings[str(k)] = str(psr)

    metadata = vars(args)

    final_save_data = {'results' : predicted_source_rankings, 'dtypes' : original_dtypes} | metadata

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, args.target_lang + '.jsonl'), 'wb') as f:
        f.write(orjson.dumps(final_save_data, option=orjson.OPT_APPEND_NEWLINE))


    # z = np.dtype([("", "<f8")])
    # a = z[0].type("104.086845") 


