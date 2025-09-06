import argparse
import faiss
import math
import numpy as np
import os
import time
import logging
from tqdm import tqdm
import orjson
from datetime import datetime
import itertools

LETTER_CODES = {"am": "amh", "bs": "bos", "vi": "vie", "wa": "wln", "eu": "eus", "so": "som", "el": "ell", "aa": "aar", "or": "ori", "sm": "smo", "gn": "grn", "mi": "mri", "pi": "pli", "ps": "pus", "ms": "msa", "sa": "san", "ko": "kor", "sd": "snd", "hz": "her", "ks": "kas", "fo": "fao", "iu": "iku", "tg": "tgk", "dz": "dzo", "ar": "ara", "fa": "fas", "es": "spa", "my": "mya", "mg": "mlg", "st": "sot", "gu": "guj", "uk": "ukr", "lv": "lav", "to": "ton", "nv": "nav", "kl": "kal", "ka": "kat", "yi": "yid", "pl": "pol", "ht": "hat", "lu": "lub", "fr": "fra", "ia": "ina", "lt": "lit", "om": "orm", "qu": "que", "no": "nor", "sr": "srp", "br": "bre", "rm": "roh", "io": "ido", "gl": "glg", "nb": "nob", "ng": "ndo", "ts": "tso", "nr": "nbl", "ee": "ewe", "bo": "bod", "mt": "mlt", "ta": "tam", "et": "est", "yo": "yor", "tw": "twi", "sl": "slv", "su": "sun", "gv": "glv", "lo": "lao", "af": "afr", "sg": "sag", "sv": "swe", "ne": "nep", "ie": "ile", "bm": "bam", "sc": "srd", "sw": "swa", "nn": "nno", "ho": "hmo", "ak": "aka", "ab": "abk", "ti": "tir", "fy": "fry", "cr": "cre", "sh": "hbs", "ny": "nya", "uz": "uzb", "as": "asm", "ky": "kir", "av": "ava", "ig": "ibo", "zh": "zho", "tr": "tur", "hu": "hun", "pt": "por", "fj": "fij", "hr": "hrv", "it": "ita", "te": "tel", "rw": "kin", "kk": "kaz", "hy": "hye", "wo": "wol", "jv": "jav", "oc": "oci", "kn": "kan", "cu": "chu", "ln": "lin", "ha": "hau", "ru": "rus", "pa": "pan", "cv": "chv", "ss": "ssw", "ki": "kik", "ga": "gle", "dv": "div", "vo": "vol", "lb": "ltz", "ce": "che", "oj": "oji", "th": "tha", "ff": "ful", "kv": "kom", "tk": "tuk", "kr": "kau", "bg": "bul", "tt": "tat", "ml": "mal", "tl": "tgl", "mr": "mar", "hi": "hin", "ku": "kur", "na": "nau", "li": "lim", "nl": "nld", "nd": "nde", "os": "oss", "la": "lat", "bn": "ben", "kw": "cor", "id": "ind", "ay": "aym", "xh": "xho", "zu": "zul", "cs": "ces", "sn": "sna", "de": "deu", "co": "cos", "sk": "slk", "ug": "uig", "rn": "run", "he": "heb", "ba": "bak", "ro": "ron", "be": "bel", "ca": "cat", "kj": "kua", "ja": "jpn", "ch": "cha", "ik": "ipk", "bi": "bis", "an": "arg", "cy": "cym", "tn": "tsn", "mk": "mkd", "ve": "ven", "eo": "epo", "kg": "kon", "km": "khm", "se": "sme", "ii": "iii", "az": "aze", "en": "eng", "ur": "urd", "za": "zha", "is": "isl", "mh": "mah", "mn": "mon", "sq": "sqi", "lg": "lug", "gd": "gla", "fi": "fin", "ty": "tah", "da": "dan", "si": "sin", "ae": "ave", "alb": "sqi", "arm": "hye", "baq": "eus", "tib": "bod", "bur": "mya", "cze": "ces", "chi": "zho", "wel": "cym", "ger": "deu", "dut": "nld", "gre": "ell", "per": "fas", "fre": "fra", "geo": "kat", "ice": "isl", "mac": "mkd", "mao": "mri", "may": "msa", "rum": "ron", "slo": "slk"}


def load_hidden_representations_from_dir(load_dir, limiter, layer, file_contains_all_hidden_reps, strip_first_last=True, model_hidden_dim=768):

    """
    Note: getting the embedding layer representation from "hidden_states" includes positional and segment 
    embeddings (when applicable).

    Therefore, they will be *different* for the same input id! 
    Sanity check: the representation for the [CLS] token is equal across all inputs! (always pos 0)
    If we want them to be the same, then need to explicilty get the output of the embedding layer when extracting.

    Need to experiment: is it beneficial to keep position embeddings or not?  
    
    """


    layer_string = f'layer_all' if file_contains_all_hidden_reps else f'layer_{layer}'

    all_files = [f for f in os.listdir(load_dir) if '.npy' in f and layer_string in f]
    found_indices = list(set([int(x.split('_')[0]) for x in all_files]))
    found_indices.sort()

    all_indices = np.empty(0, dtype=int)
    all_input_ids = np.empty(0, dtype=int)
    all_hidden_reps = np.empty((0,model_hidden_dim))

    found_indices = found_indices[:limiter] if limiter is not None else found_indices

    for idx in tqdm(found_indices):

        hidden_file = os.path.join(load_dir, f'{idx}_hidden_{layer_string}.npy')
        input_ids_file = os.path.join(load_dir, f'{idx}_input_ids_{layer_string}.npy')

        input_ids = np.load(input_ids_file)
        input_ids = input_ids.squeeze(axis=0)

        if strip_first_last:
            input_ids = input_ids[1:-1]

        all_input_ids = np.concatenate([all_input_ids, input_ids], axis=0)
        
        hidden_reps = np.load(hidden_file)
        hidden_reps = hidden_reps[layer] if file_contains_all_hidden_reps else hidden_reps

        if strip_first_last:
            hidden_reps = hidden_reps[1:-1]

        all_hidden_reps = np.concatenate([all_hidden_reps, hidden_reps], axis=0)

        indices = np.repeat(idx, len(input_ids))

        # if strip_first_last:
        #     indices = indices[1:-1]
        
        all_indices = np.concatenate([all_indices, indices], axis=0)

    logging.info(f'Loaded {len(all_hidden_reps)} hidden representations from {len(found_indices)} input indices in dir {load_dir}.')
    return all_indices, all_input_ids, all_hidden_reps


def process_ud_directories(directory_list):
    
    isos = []
    langs = []
    directories = {}


    for directory in directory_list:
        try:

            if os.path.isdir(directory):

                bname = os.path.basename(directory)
                iso = bname.split('_')[0]
                
                isos.append(iso)
                langs.append(bname)
                directories[bname] = directory
            else:
                logging.error(f'Error loading directory: {directory}')

        except:
            logging.error(f'Caught exception in data loading. Did not load: {directory}')

    return isos, langs, directories


def process_ner_directories(directory_list):
    
    isos = []
    langs = []
    directories = {}

    for directory in directory_list:
        
        try:

            if os.path.isdir(directory):
                iso = directory.split('/')[-1]
                directory_split = directory.split('/')[-2]
                lang_name = f'{iso}_{directory_split}'
                
                isos.append(iso)
                langs.append(lang_name)
                directories[lang_name] = directory
            else:
                logging.error(f'Error loading directory: {directory}')

        except:
            logging.error(f'Caught exception in data loading. Did not load: {directory}')

    return isos, langs, directories

def process_bible_directories(directory_list):

    isos = []
    langs = []
    directories = {}

    for directory in directory_list:
        try:

            if os.path.isdir(directory):
                bname = os.path.basename(directory)
                iso = bname.split('-')[0]
                
                isos.append(iso)
                langs.append(bname)
                directories[bname] = directory
            else:
                logging.error(f'Error loading directory: {directory}')

        except:
            logging.error(f'Caught exception in data loading. Did not load: {directory}')

    logging.info(f'Finished loading bible directories. Loaded {len(isos)} sets of representations.')
    logging.info(f'Due to bible loading, will check langs for duplicates')

    unique_isos = []
    unique_langs = []
    unique_directories = {}

    for i, lang in enumerate(langs):
        if lang in unique_langs:
            logging.info(f'Skipping {lang} due to duplicate')
        else:
            logging.info(f'Added lang {lang} to unique langs')
            unique_isos.append(isos[i])
            unique_langs.append(langs[i])
            unique_directories[langs[i]] = directories[langs[i]]

    logging.info(f'Finished de-duplicating with final number of loaded langs: {len(unique_langs)}')

    return unique_isos, unique_langs, unique_directories



def setup_logging(log_dir, experiment_name, run_name, shard_idx):
    
    intended_log_dir = os.path.join(log_dir, experiment_name, run_name)

    date_time = datetime.now().strftime("%Y%d%m-%H%M%S")

    if not os.path.exists(intended_log_dir):
        os.makedirs(intended_log_dir, exist_ok=True)

    # Setup logging
    logging.root.handlers = []
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}/run-{}-shard-{}.log".format(intended_log_dir, date_time, shard_idx)),
            logging.StreamHandler()
        ]
    )



if __name__ == '__main__':

    """
        Description of output:

        Each of the {source,target} matrices is a map from our index to various data. Here, our "index" represents each token which we have seen from the data (with repeats)
        Note that we need repeats: embeddings for the same subword may be the same depending on sentence position, and of course hidden representations will change depending on differing contexts


            target_hidden_reps: (280, 768)  # Map each token to it's hidden representation
            target_indices: (280,)          # Map each token to the line index from the corpus
            target_input_ids: (280,)        # Map each token to its input id given to the model
            target_langs: (280,)            # Map each token to the language of the corpus it was taken from.

            
            source_hidden_reps: (1094, 768)
            source_indices: (1094,)
            source_input_ids: (1094,)
            source_langs: (1094,)

            distances: (280, 10)            # For each target token, its top-k nearest neighbors
            neighbors_indices: (280, 10)    # For each target token, the index of its nearest neighbors


        Examples:

        To get all tokens from the first line of the target corpus:
            np.where(target_indices == 0)

        To get all input ids from the first line of the target corpus:
            target_input_ids[np.where(target_indices == 0)]

        To get the nearest neighbors of the first token of the first line:
            fl = target_input_ids[np.where(target_indices == 0)]
            ft = fl[0]

            neighbors = neighbors_indices[ft]

        To get the input ids of the nearest neighbors:
            source_input_ids[neighbors]

        To get the languages of the nearest neighbors:
            source_langs[neighbors]


    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--target_directories', type=str, nargs='+')
    parser.add_argument('--source_directories', type=str, nargs='+')
    
    parser.add_argument('--source_directory_type', default='ud')
    parser.add_argument('--target_directory_type', default='ud')

    parser.add_argument('--root_output_dir')
    parser.add_argument('--log_dir', default='logs/nn_logs/')
    parser.add_argument('--experiment_name')
    parser.add_argument('--run_name', nargs='+', type=str)

    parser.add_argument('--num_shards', default=1, type=int)
    parser.add_argument('--shard_idx', default=0, type=int)

    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--limiter', type=int, default=None)
    parser.add_argument('--start_at_i', type=int, default=-1)
    parser.add_argument('--remove_target_from_source_langs', default=False, action='store_const', const=True)
    parser.add_argument('--load_target_languages_independently', default=False, action='store_const', const=True, help='Load target language data only when needed. Helpful to save memory when the pool of source langs and target langs are mutually exclusive.')
    parser.add_argument('--extract_layer_hidden_rep_from_all', default=False, action='store_const', const=True, help='Set flag to true if layer flag was not used when extracting. Otherwise will look for file with the specified layer in the name.')
    parser.add_argument('--check_already_trained', action='store_const', const=True, default=False)
    args = parser.parse_args()

    # ---- Setup Directories ---- #

    """
    run_name behavior:
        run_name is a list of arguments
        we take the root output dir and join it with all elements of run_name, where each element is a subdirectory
        output_dir = root_output_dir/run_name[0]/.../run_name[n]

        however we are processing multiple target languages, so for the final step of saving, we will join the target_lang to the path:
        final output dir = root_output_dir/run_name[0]/.../run_name[n]/target_lang

        for logging, we will log to:
            log_dir/run_name[0]/.../run_name[n]/run-{datetime}.log
    """
    run_name = args.run_name[0] if len(args.run_name) == 1 else '/'.join(args.run_name[:-1]) + f'/{os.path.splitext(args.run_name[-1])[0]}'

    setup_logging(args.log_dir, args.experiment_name, run_name, args.shard_idx)

    # ---- Deduplicate target languages ---- #
    logging.info(f'Given {len(args.target_directories)} target directories.')
    unique_target_directories = []

    for td in args.target_directories:
        if td not in unique_target_directories:
            logging.info(f'Adding {td} to target directories.')
            unique_target_directories.append(td)
        else:
            logging.info(f'Skipping {td} due to duplicate.')

    logging.info(f'Num unique target directories: {len(unique_target_directories)}')


    # ---- Shard Target Langauges ---- # 

    logging.info('---- Sharding data ----')
    
    shard_size = math.ceil(len(unique_target_directories) / args.num_shards)

    logging.info(f'Sharding into {args.num_shards} shards, which gives a shard size of {shard_size}')
    logging.info(f'Given shard idx: {args.shard_idx}')


    shards = [unique_target_directories[shard_size * i:shard_size*i+shard_size] for i in range(args.num_shards)]

    logging.info(f'Corresponding to target directories: {shards[args.shard_idx]}')

    logging.info(f'Given {len(args.source_directories)} source directories.')
    logging.info(f'Source directories: {args.source_directories}')

    target_dirs_from_shard_to_load = shards[args.shard_idx]


    # ---- Load Data ---- #


    # iso2dir functions should take in a list of directories (one for each dataset) and extract three items: list of isos (not unique), list of langs (unique identifier), map from lang to load directory.
    
    iso2dir_factory = {
        'ud' : process_ud_directories,
        'ner' : process_ner_directories,
        'bible' : process_bible_directories
    }

    # Here, langs are unique identifiers, however the ISOs may be duplicated
    target_isos, target_langs, target_directories = iso2dir_factory[args.target_directory_type](target_dirs_from_shard_to_load)
    source_isos, source_langs, source_directories = iso2dir_factory[args.source_directory_type](args.source_directories)

    assert len(set(target_langs)) == len(target_langs)
    assert len(set(source_langs)) == len(source_langs)

    all_langs = target_langs + source_langs
    all_dirs = target_directories | source_directories

    assert len(target_langs) == len(target_isos)
    assert len(source_langs) == len(source_isos)

    target_lang2iso = {lang : iso for lang, iso in zip(target_langs, target_isos)}
    source_lang2iso = {lang : iso for lang, iso in zip(source_langs, source_isos)}

    if args.check_already_trained: #Do a preliminary check to see if all target languages have saved outputs. 
        logging.info(f'Running preliminary check to see if all target languages have saved outputs.')
        already_trained = []
        for i, target_lang in enumerate(target_langs):

            if args.start_at_i > 0 and i < args.start_at_i:
                logging.info(f'Skipping target language {target_lang} due to args.start_at_i set to i={args.start_at_i}')
                continue

            specific_run_name = os.path.join(run_name, target_lang)
            output_dir = os.path.join(args.root_output_dir, args.experiment_name, specific_run_name)

            logging.info(f'Checking target lang {target_lang} at directory {output_dir}')

            if os.path.exists(os.path.join(output_dir, 'metadata.jsonl')):
                logging.info(f'Output data for target language {target_lang} exists at output directory {output_dir}. Continuing')
                already_trained.append(True)

            else:
                logging.info(f'Output data for target language {target_lang} does not exist.')
                already_trained.append(False)

        assert len(target_langs) == len(already_trained)

        if not all(already_trained):
            logging.info(f'There are target languages which do not have saved data. Will continue loading')
            for target_lang, at in zip(target_langs, already_trained):
                s = "save exists" if at else "save not found"
                logging.info(f'{target_lang:40s}{s:>30s}')

        else:
            logging.info(f'All target languages have saved data. Will exit.')
            quit(0)

    if args.load_target_languages_independently:
        # Do not load target language data now.
        all_data = {lang : load_hidden_representations_from_dir(directory, limiter=args.limiter, layer=args.layer, file_contains_all_hidden_reps=args.extract_layer_hidden_rep_from_all) for lang,directory in tqdm(source_directories.items())}
    else:
        all_data = {lang : load_hidden_representations_from_dir(directory, limiter=args.limiter, layer=args.layer, file_contains_all_hidden_reps=args.extract_layer_hidden_rep_from_all) for lang,directory in tqdm(all_dirs.items())}

    logging.info(f'source langs which were loaded (n={len(source_lang2iso)}): {source_lang2iso}')


    logging.info(f'All target langs in this run:')
    for tl in target_langs:
        logging.info(tl)

    logging.info(f'Total Number of target langs: {len(target_langs)}')
    
    
    for i, target_lang in enumerate(target_langs):

        logging.info('\n\n' + '*'*40 + f'Processing Target Language: {target_lang}' + '*'*40 + '\n\n')

        if args.start_at_i > 0 and i < args.start_at_i:
            logging.info(f'Skipping target language {target_lang} due to args.start_at_i set to i={args.start_at_i}')
            continue

        specific_run_name = os.path.join(run_name, target_lang)
        output_dir = os.path.join(args.root_output_dir, args.experiment_name, specific_run_name)

        logging.info(f'Setting output directory to: {output_dir}/')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.check_already_trained and os.path.exists(os.path.join(output_dir, 'metadata.jsonl')):
            logging.info(f'Output data for target language {target_lang} exists at output directory {output_dir}. Skipping')
            continue

        logging.info(f'Starting to process target language {target_lang} num: #{i}')

        if args.load_target_languages_independently:

            #Load the target langauge representations now
            target_indices, target_input_ids, target_hidden_reps = load_hidden_representations_from_dir(
                load_dir = target_directories[target_lang],
                limiter = args.limiter,
                layer = args.layer,
                file_contains_all_hidden_reps=args.extract_layer_hidden_rep_from_all
            )
        
        else:
            # Target language representations already loaded, get them from all data
            target_indices, target_input_ids, target_hidden_reps = all_data[target_lang]


        target_langs = np.repeat(target_lang, len(target_indices))


        if args.remove_target_from_source_langs:
            source_pool = []
            removed_pool = []

            for sl in source_langs:

                source_iso_check = source_lang2iso[sl]
                target_iso_check = target_lang2iso[target_lang]

                potential_source_iso_three = LETTER_CODES.get(source_iso_check, -1)
                potential_target_iso_three = LETTER_CODES.get(target_iso_check, -2)

                logging.info(f'Checking for source ISOs equal to the target iso.')
                logging.info(f'\tThe four potential iso codes: {source_iso_check}, {target_iso_check}, {potential_source_iso_three}, {potential_target_iso_three}. All 4 must not be equal to pass.')

                check_iso_combinations = list(itertools.product([source_iso_check, potential_source_iso_three], [target_iso_check, potential_target_iso_three]))
                logging.info(f'\tIso combinations: {[x for x in check_iso_combinations]}')
                
                truth_values = [iso_comb[0] != iso_comb[1] for iso_comb in check_iso_combinations]
                logging.info(f'\tTruth values: {truth_values}')
                
                if all(truth_values):
                    source_pool.append(sl)
                else:
                    removed_pool.append(sl)
            logging.info(f'Source langs which were removed: {removed_pool}')
            logging.info(f'Source languages which were kept: {source_pool}')

        else:
            source_pool = source_langs

        logging.info(f'Loaded {len(source_pool)} total source languages, selected from {len(source_langs)} options.')

        source_data = [all_data[source_lang] for source_lang in source_pool]

        all_source_indices = np.concatenate([x[0] for x in source_data])
        all_source_input_ids = np.concatenate([x[1] for x in source_data])
        all_source_hidden_reps = np.concatenate([x[2] for x in source_data])
        all_source_langs = np.concatenate([np.repeat(lang, len(x[0])) for lang, x in zip(source_pool, source_data)])    

        index = faiss.IndexFlatIP(all_source_hidden_reps.shape[1])
        index.add(all_source_hidden_reps)

        distances, neighbors_indices = index.search(target_hidden_reps, k=args.k)

        arg_dict = vars(args)
        arg_dict['source_pool'] = source_pool
        arg_dict['target_lang'] = target_lang

        save_dictionary = {
            'source_indices' : all_source_indices,
            'source_input_ids' : all_source_input_ids,
            # 'source_hidden_reps' : all_source_hidden_reps,
            'source_langs' : all_source_langs,
            'distances': distances,
            'neighbors_indices' : neighbors_indices,
            'target_indices' : target_indices,
            'target_input_ids' : target_input_ids,
            # 'target_hidden_reps' : target_hidden_reps,
            'target_langs' : target_langs,
            'metadata' : arg_dict
        }

        logging.info(f'done calculating nn for target lang: {target_lang}')

        for k,v in save_dictionary.items():
            if isinstance(v, np.ndarray):
                np.save(os.path.join(output_dir, k + '.npy'), v)

            elif isinstance(v, dict):
                with open(os.path.join(output_dir, f'{k}.jsonl'), 'wb') as f:
                    f.write(orjson.dumps(v))

        logging.info(f'saved nns for target lang: {target_lang} #{i}')




   






    

    
