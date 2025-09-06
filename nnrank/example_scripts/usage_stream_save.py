from nnrank.ranker import NNRanker
import os
from itertools import combinations

def extract_filename(fpath):

    return os.path.splitext(os.path.basename(fpath))[0]


if __name__ == '__main__':

    source_datasets_path = '/projects/abeb4417/multilingual_analysis_pub/nnrank/mwe_bibles/mwe_data/source_pool'
    target_datasets_path = '/projects/abeb4417/multilingual_analysis_pub/nnrank/mwe_bibles/mwe_data/target_datasets'

    source_datasets = [os.path.join(source_datasets_path, f) for f in os.listdir(source_datasets_path)]
    target_datasets = [os.path.join(target_datasets_path, f) for f in os.listdir(target_datasets_path)]

    print(f'Found {len(source_datasets)} source datasets')
    print(f'Found {len(target_datasets)} target datasets')

    source_dict = {extract_filename(f) : f for f in source_datasets}
    target_dict = {extract_filename(f) : f for f in target_datasets}

    source_names = list(source_dict.keys())
    target_names = list(target_dict.keys())

    ranker = NNRanker('bert-base-multilingual-cased')

    ranker.load_text_data(source_dict | target_dict, limiter=25)
    ranker.set_source_languages(source_names)

    # Extract all hidden representations and save directly to disk (if GPU resources are limited)
    ranker.extract_hidden_representations(keys=source_names+target_names, keep_in_memory=False, stream_save=True, save_every=5, root_output_dir='example_outputs/stream_save/')

    # In a different script, re-load saved representations
    new_ranker = NNRanker('bert-base-multilingual-cased')
    new_ranker.set_source_languages(source_names)
    new_ranker.load_hidden_representations(keys=source_names, root_load_dir='example_outputs/stream_save/')

    rankings = {}

    for target_name in target_names:
        # Extract target datasets one by one to prevent out of memory
        new_ranker.load_hidden_representations(keys=[target_name], root_load_dir='example_outputs/stream_save/')
        rankings[target_name] = new_ranker.rank([target_name])

    for target, ranking in rankings.items():
        print(f'Target: {target} Ranking: {ranking}')


    """
    Target: nno-x-bible-2011-v1 Ranking: {'nno-x-bible-2011-v1': [('ess', 1602), ('rus-x-bible-kulakov-v1', 1196), ('deu-x-bible-freebible-v1', 317), ('eng', 232), ('grc-x-bible-textusreceptusVAR1-v1', 43), ('cop-x-bible-bohairic-v1', 0)]}
    Target: frr Ranking: {'frr': [('eng', 2627), ('deu-x-bible-freebible-v1', 1467), ('ess', 117), ('rus-x-bible-kulakov-v1', 62), ('grc-x-bible-textusreceptusVAR1-v1', 27), ('cop-x-bible-bohairic-v1', 0)]}
    Target: grc-x-bible-textusreceptusVAR1-v1 Ranking: {'grc-x-bible-textusreceptusVAR1-v1': [('grc-x-bible-textusreceptusVAR1-v1', 6967), ('ess', 2), ('rus-x-bible-kulakov-v1', 1), ('cop-x-bible-bohairic-v1', 0), ('eng', 0), ('deu-x-bible-freebible-v1', 0)]}
    Target: fij-x-bible-fij-v1 Ranking: {'fij-x-bible-fij-v1': [('grc-x-bible-textusreceptusVAR1-v1', 3072), ('rus-x-bible-kulakov-v1', 1249), ('ess', 784), ('eng', 114), ('deu-x-bible-freebible-v1', 55), ('cop-x-bible-bohairic-v1', 26)]}
    """