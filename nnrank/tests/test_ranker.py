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

    memory_ranker = NNRanker('bert-base-multilingual-cased')
    memory_ranker_to_save = NNRanker('bert-base-multilingual-cased')
    stream_save_ranker = NNRanker('bert-base-multilingual-cased')
    disk_save_ranker = NNRanker('bert-base-multilingual-cased')

    for ranker in [memory_ranker, memory_ranker_to_save, stream_save_ranker, disk_save_ranker]:

        ranker.load_text_data(source_dict | target_dict, 25)
        ranker.set_source_languages(source_dict.keys())

    test_keys = list(source_dict.keys()) + list(target_dict.keys())
    print(test_keys)

    memory_ranker.extract_hidden_representations(test_keys, keep_in_memory=True, stream_save=False, save_to_disk=False)
    memory_ranker_to_save.extract_hidden_representations(test_keys, keep_in_memory=True, stream_save=False, save_to_disk=False)
    stream_save_ranker.extract_hidden_representations(test_keys, keep_in_memory=False, stream_save=True, save_every=5, save_to_disk=False, root_output_dir='test_outs/stream_save/')
    disk_save_ranker.extract_hidden_representations(test_keys, keep_in_memory=False, stream_save=False, save_to_disk=True, root_output_dir='test_outs/disk_save/')

    for ranker, rname in zip([memory_ranker, memory_ranker_to_save, stream_save_ranker, disk_save_ranker], ['memory_ranker', 'memory_ranker_to_save', 'stream_save_ranker', 'disk_save_ranker']):
        print(f'{rname:<25s} {ranker.hidden_representations.keys()}')

    memory_ranker_to_save.save_hidden_representations(test_keys, root_output_dir='test_outs/memory_ranker_to_save/')
    memory_ranker_to_save.hidden_representations = {}
    memory_ranker_to_save.load_hidden_representations(test_keys, 'test_outs/memory_ranker_to_save/')
    stream_save_ranker.load_hidden_representations(test_keys, 'test_outs/stream_save/')
    disk_save_ranker.load_hidden_representations(test_keys, 'test_outs/disk_save/')

    for (r1, rname1), (r2, rname2) in combinations(zip([memory_ranker, memory_ranker_to_save, stream_save_ranker, disk_save_ranker], ['memory_ranker', 'memory_ranker_to_save', 'stream_save_ranker', 'disk_save_ranker']), r=2):
        for lang in r1.hidden_representations:
            for key in r1.hidden_representations[lang]:
                
                print(f'{rname1:>25s}{rname2:>25s}{lang:>50s}{key:>25s} {(r1.hidden_representations[lang][key] == r2.hidden_representations[lang][key]).all()!s:>50}')


    ranking_results = {}
    for ranker, rname in zip([memory_ranker, memory_ranker_to_save, stream_save_ranker, disk_save_ranker], ['memory_ranker', 'memory_ranker_to_save', 'stream_save_ranker', 'disk_save_ranker']):
        ranking_results[rname] = ranker.rank(target_dict.keys(), k=5)

    for (r1, rname1), (r2, rname2) in combinations(zip([memory_ranker, memory_ranker_to_save, stream_save_ranker, disk_save_ranker], ['memory_ranker', 'memory_ranker_to_save', 'stream_save_ranker', 'disk_save_ranker']), r=2):
        print(f'{rname1:>25s}{rname2:>25s} {ranking_results[rname1] == ranking_results[rname2]!s:>125}')



