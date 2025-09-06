import os
import orjson
import collections
import logging
import numpy as np
from tqdm import tqdm
from ma_utils.custom_datasets.pos_datasets import ConllPOSDataset
from transformers import AutoTokenizer
# import ma_utils.pos_lang_maps as pos_lang_maps



# Taken from langrank_predict.py
LETTER_CODES = {"am": "amh", "bs": "bos", "vi": "vie", "wa": "wln", "eu": "eus", "so": "som", "el": "ell", "aa": "aar", "or": "ori", "sm": "smo", "gn": "grn", "mi": "mri", "pi": "pli", "ps": "pus", "ms": "msa", "sa": "san", "ko": "kor", "sd": "snd", "hz": "her", "ks": "kas", "fo": "fao", "iu": "iku", "tg": "tgk", "dz": "dzo", "ar": "ara", "fa": "fas", "es": "spa", "my": "mya", "mg": "mlg", "st": "sot", "gu": "guj", "uk": "ukr", "lv": "lav", "to": "ton", "nv": "nav", "kl": "kal", "ka": "kat", "yi": "yid", "pl": "pol", "ht": "hat", "lu": "lub", "fr": "fra", "ia": "ina", "lt": "lit", "om": "orm", "qu": "que", "no": "nor", "sr": "srp", "br": "bre", "rm": "roh", "io": "ido", "gl": "glg", "nb": "nob", "ng": "ndo", "ts": "tso", "nr": "nbl", "ee": "ewe", "bo": "bod", "mt": "mlt", "ta": "tam", "et": "est", "yo": "yor", "tw": "twi", "sl": "slv", "su": "sun", "gv": "glv", "lo": "lao", "af": "afr", "sg": "sag", "sv": "swe", "ne": "nep", "ie": "ile", "bm": "bam", "sc": "srd", "sw": "swa", "nn": "nno", "ho": "hmo", "ak": "aka", "ab": "abk", "ti": "tir", "fy": "fry", "cr": "cre", "sh": "hbs", "ny": "nya", "uz": "uzb", "as": "asm", "ky": "kir", "av": "ava", "ig": "ibo", "zh": "zho", "tr": "tur", "hu": "hun", "pt": "por", "fj": "fij", "hr": "hrv", "it": "ita", "te": "tel", "rw": "kin", "kk": "kaz", "hy": "hye", "wo": "wol", "jv": "jav", "oc": "oci", "kn": "kan", "cu": "chu", "ln": "lin", "ha": "hau", "ru": "rus", "pa": "pan", "cv": "chv", "ss": "ssw", "ki": "kik", "ga": "gle", "dv": "div", "vo": "vol", "lb": "ltz", "ce": "che", "oj": "oji", "th": "tha", "ff": "ful", "kv": "kom", "tk": "tuk", "kr": "kau", "bg": "bul", "tt": "tat", "ml": "mal", "tl": "tgl", "mr": "mar", "hi": "hin", "ku": "kur", "na": "nau", "li": "lim", "nl": "nld", "nd": "nde", "os": "oss", "la": "lat", "bn": "ben", "kw": "cor", "id": "ind", "ay": "aym", "xh": "xho", "zu": "zul", "cs": "ces", "sn": "sna", "de": "deu", "co": "cos", "sk": "slk", "ug": "uig", "rn": "run", "he": "heb", "ba": "bak", "ro": "ron", "be": "bel", "ca": "cat", "kj": "kua", "ja": "jpn", "ch": "cha", "ik": "ipk", "bi": "bis", "an": "arg", "cy": "cym", "tn": "tsn", "mk": "mkd", "ve": "ven", "eo": "epo", "kg": "kon", "km": "khm", "se": "sme", "ii": "iii", "az": "aze", "en": "eng", "ur": "urd", "za": "zha", "is": "isl", "mh": "mah", "mn": "mon", "sq": "sqi", "lg": "lug", "gd": "gla", "fi": "fin", "ty": "tah", "da": "dan", "si": "sin", "ae": "ave", "alb": "sqi", "arm": "hye", "baq": "eus", "tib": "bod", "bur": "mya", "cze": "ces", "chi": "zho", "wel": "cym", "ger": "deu", "dut": "nld", "gre": "ell", "per": "fas", "fre": "fra", "geo": "kat", "ice": "isl", "mac": "mkd", "mao": "mri", "may": "msa", "rum": "ron", "slo": "slk"}


def create_ud_indexes(ud_data_dir='/projects/abeb4417/multilingual_analysis/data/pos_ud_data/ud-treebanks-v2.15/',
                      output_dir='/projects/abeb4417/multilingual_analysis/ma_utils/pos_file_iso_maps/'):


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    ud_folders = os.listdir(ud_data_dir)

    langrank_train_code_to_filename = {}
    langrank_train_code_to_dir = {}

    langrank_supported_eval_iso_codes = list(LETTER_CODES.keys())
    langrank_supported_eval_iso_codes.sort()

    langrank_supported_dev_folders = []
    langrank_supported_test_folders = []

    langrank_indexed_pos_sets = np.load('/projects/abeb4417/multilingual_analysis/langrank/langrank/indexed/POS/ud.npy', allow_pickle=True).item()
    langrank_source_datasets = list(langrank_indexed_pos_sets.keys())

    ud_test_to_dev_map = get_ud_test_to_dev_map(ud_data_dir)

    splits = ['train', 'dev', 'test']
    num_examples = {}
    for split in splits:
        num_examples[split] = find_ud_num_lines(ud_data_dir, split)

    folder_to_file_map = {split : {} for split in splits}


    for folder in ud_folders:
        lang_name, lang_domain = folder.replace('UD_', '').split('-')
        folder_files = os.listdir(os.path.join(ud_data_dir, folder))

        for f in folder_files:
            if '.conllu' in f:
                data_file = f
                break

        lang_code = data_file.split('_')[0]
        domain_lower = data_file.split('_')[1].split('-')[0] 

        for f in folder_files:
            if '.conllu' in f:

                fbasename = os.path.splitext(f)[0]
                
                if 'train' in f:
                    langrank_string_to_test = f'datasets/pos/{lang_code}_{domain_lower}'

                    if langrank_string_to_test in langrank_source_datasets:
                        langrank_train_code_to_filename[f'datasets/pos/{lang_code}_{domain_lower}'] = os.path.splitext(f)[0]
                        langrank_train_code_to_dir[f'datasets/pos/{lang_code}_{domain_lower}'] = folder
                        folder_to_file_map['train'][folder] = fbasename

                if 'dev' in f:
                    if lang_code in langrank_supported_eval_iso_codes:
                        langrank_supported_dev_folders.append(folder)
                        folder_to_file_map['dev'][folder] = fbasename

                if 'test' in f:
                    if lang_code in langrank_supported_eval_iso_codes and fbasename in ud_test_to_dev_map:
                        langrank_supported_test_folders.append(folder)
                        folder_to_file_map['test'][folder] = fbasename


    with open(os.path.join(output_dir, 'pos_lang_maps.py'), 'w') as f:
        f.write('LANGRANK_DSCODE_TO_FILE = ' + orjson.dumps(langrank_train_code_to_filename, option=orjson.OPT_INDENT_2).decode('utf-8') + '\n')


    with open(os.path.join(output_dir, 'langrank_eval_isos'), 'w') as f:
        for iso in langrank_supported_eval_iso_codes:
            f.write(f'{iso}\n')

    with open(os.path.join(output_dir, 'langrank_source_datasets_not_found'), 'w') as f:
        for lsd in langrank_source_datasets:
            if lsd not in langrank_train_code_to_dir:
                f.write(f'{lsd}\n')

    for threshold, name in [(100, 'all'), (750, 'medium'), (2000, 'large')]:

        with open(os.path.join(output_dir, f'langrank_dev_{name}_folders'), 'w') as f:
            for folder in langrank_supported_dev_folders:
                if num_examples['dev'][folder] > threshold:
                    f.write(f'{folder}\n')

        with open(os.path.join(output_dir, f'langrank_dev_{name}_fnames'), 'w') as f:
            for folder in langrank_supported_dev_folders:
                if num_examples['dev'][folder] > threshold:
                    f.write(f'{folder_to_file_map["dev"][folder]}\n')

        with open(os.path.join(output_dir, f'langrank_test_{name}_folders'), 'w') as f:
            for folder in langrank_supported_test_folders:
                if num_examples['test'][folder] > threshold:
                    f.write(f'{folder}\n')

        with open(os.path.join(output_dir, f'langrank_test_{name}_fnames'), 'w') as f:
            for folder in langrank_supported_test_folders:
                if num_examples["test"][folder] > threshold:
                    f.write(f'{folder_to_file_map["test"][folder]}\n')

    for threshold, name in [(500, 'all'), (7500, 'medium'), (15000, 'large')]:


        with open(os.path.join(output_dir, f'langrank_train_{name}_folders'), 'w') as f:
            for folder in langrank_train_code_to_dir.values():
                if num_examples['train'][folder] > threshold:
                    f.write(f'{folder}\n')

        with open(os.path.join(output_dir, f'langrank_train_{name}_fnames'), 'w') as f:
            for folder in langrank_train_code_to_dir.values():
                if num_examples['train'][folder] > threshold:
                    f.write(f'{folder_to_file_map["train"][folder]}\n')

                
                
"""
Calculate number of training examples for each UD train set.

"""
def find_ud_num_train_lines(ud_dir):
    
    records = []
    
    ud_dirs = os.listdir(ud_dir)
    for ud_set in tqdm(ud_dirs):
        p = os.path.join(ud_dir, ud_set)
        for fname in os.listdir(p):
            if 'train' in fname and '.txt' in fname:
                full_path = os.path.join(p, fname)
                with open(full_path, 'r') as f:
                    records.append({
                        'train_file' : fname.replace('.txt', ''),
                        'num_train_examples' : len(f.readlines())
                    })

  
    return records


def find_ud_num_lines(ud_dir, split='train'):
    
    folder_to_examples_map = {}
    
    ud_dirs = os.listdir(ud_dir)
    for ud_set in tqdm(ud_dirs):
        p = os.path.join(ud_dir, ud_set)
        for fname in os.listdir(p):
            if split in fname and '.txt' in fname:
                full_path = os.path.join(p, fname)
                with open(full_path, 'r') as f:
                    folder_to_examples_map[ud_set] = len(f.readlines())
                

  
    return folder_to_examples_map


"""
Return a map from test file names to dev file names, if they exist

"""
def get_ud_test_to_dev_map(ud_dir):
    
    test_to_dev_map = {}
    
    ud_dirs = os.listdir(ud_dir)
    for ud_set in tqdm(ud_dirs):
        p = os.path.join(ud_dir, ud_set)
        test_file = None
        dev_file = None
        for fname in os.listdir(p):
            if 'test' in fname and '.conllu' in fname:
                test_file = os.path.splitext(fname)[0]
            elif 'dev' in fname and '.conllu' in fname:
                dev_file = os.path.splitext(fname)[0]
        
        if test_file and dev_file:
            test_to_dev_map[test_file] = dev_file

  
    return test_to_dev_map

if __name__ == '__main__':

    create_ud_indexes()