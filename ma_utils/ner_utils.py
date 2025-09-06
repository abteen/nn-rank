import os
import orjson
import collections
import logging
import numpy as np
import lang2vec.lang2vec as l2v

LETTER_CODES = {"am": "amh", "bs": "bos", "vi": "vie", "wa": "wln", "eu": "eus", "so": "som", "el": "ell", "aa": "aar", "or": "ori", "sm": "smo", "gn": "grn", "mi": "mri", "pi": "pli", "ps": "pus", "ms": "msa", "sa": "san", "ko": "kor", "sd": "snd", "hz": "her", "ks": "kas", "fo": "fao", "iu": "iku", "tg": "tgk", "dz": "dzo", "ar": "ara", "fa": "fas", "es": "spa", "my": "mya", "mg": "mlg", "st": "sot", "gu": "guj", "uk": "ukr", "lv": "lav", "to": "ton", "nv": "nav", "kl": "kal", "ka": "kat", "yi": "yid", "pl": "pol", "ht": "hat", "lu": "lub", "fr": "fra", "ia": "ina", "lt": "lit", "om": "orm", "qu": "que", "no": "nor", "sr": "srp", "br": "bre", "rm": "roh", "io": "ido", "gl": "glg", "nb": "nob", "ng": "ndo", "ts": "tso", "nr": "nbl", "ee": "ewe", "bo": "bod", "mt": "mlt", "ta": "tam", "et": "est", "yo": "yor", "tw": "twi", "sl": "slv", "su": "sun", "gv": "glv", "lo": "lao", "af": "afr", "sg": "sag", "sv": "swe", "ne": "nep", "ie": "ile", "bm": "bam", "sc": "srd", "sw": "swa", "nn": "nno", "ho": "hmo", "ak": "aka", "ab": "abk", "ti": "tir", "fy": "fry", "cr": "cre", "sh": "hbs", "ny": "nya", "uz": "uzb", "as": "asm", "ky": "kir", "av": "ava", "ig": "ibo", "zh": "zho", "tr": "tur", "hu": "hun", "pt": "por", "fj": "fij", "hr": "hrv", "it": "ita", "te": "tel", "rw": "kin", "kk": "kaz", "hy": "hye", "wo": "wol", "jv": "jav", "oc": "oci", "kn": "kan", "cu": "chu", "ln": "lin", "ha": "hau", "ru": "rus", "pa": "pan", "cv": "chv", "ss": "ssw", "ki": "kik", "ga": "gle", "dv": "div", "vo": "vol", "lb": "ltz", "ce": "che", "oj": "oji", "th": "tha", "ff": "ful", "kv": "kom", "tk": "tuk", "kr": "kau", "bg": "bul", "tt": "tat", "ml": "mal", "tl": "tgl", "mr": "mar", "hi": "hin", "ku": "kur", "na": "nau", "li": "lim", "nl": "nld", "nd": "nde", "os": "oss", "la": "lat", "bn": "ben", "kw": "cor", "id": "ind", "ay": "aym", "xh": "xho", "zu": "zul", "cs": "ces", "sn": "sna", "de": "deu", "co": "cos", "sk": "slk", "ug": "uig", "rn": "run", "he": "heb", "ba": "bak", "ro": "ron", "be": "bel", "ca": "cat", "kj": "kua", "ja": "jpn", "ch": "cha", "ik": "ipk", "bi": "bis", "an": "arg", "cy": "cym", "tn": "tsn", "mk": "mkd", "ve": "ven", "eo": "epo", "kg": "kon", "km": "khm", "se": "sme", "ii": "iii", "az": "aze", "en": "eng", "ur": "urd", "za": "zha", "is": "isl", "mh": "mah", "mn": "mon", "sq": "sqi", "lg": "lug", "gd": "gla", "fi": "fin", "ty": "tah", "da": "dan", "si": "sin", "ae": "ave", "alb": "sqi", "arm": "hye", "baq": "eus", "tib": "bod", "bur": "mya", "cze": "ces", "chi": "zho", "wel": "cym", "ger": "deu", "dut": "nld", "gre": "ell", "per": "fas", "fre": "fra", "geo": "kat", "ice": "isl", "mac": "mkd", "mao": "mri", "may": "msa", "rum": "ron", "slo": "slk"}


def create_ner_indexes_1(ner_compressed_data_dir='/projects/abeb4417/data/ner/rahimi/',
                      output_dir='/projects/abeb4417/multilingual_analysis/ma_utils/ner_file_iso_maps/'):


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    langrank_supported_train_folders = []
    langrank_supported_dev_folders = []
    langrank_supported_test_folders = []

    langrank_indexed_el_sets = np.load('/projects/abeb4417/multilingual_analysis/langrank/langrank/indexed/EL/wiki.npy', allow_pickle=True).item()
    cands = []
    cands += [(key,langrank_indexed_el_sets[key]) for key in langrank_indexed_el_sets if key != "eng"]

    langs_orig = [c[0].replace('wiki_en-', '') for c in cands if c[1]['lang'] in l2v.available_learned_languages()]
    langs3 = [c[1]['lang'] for c in cands if c[1]['lang'] in l2v.available_learned_languages()]


    for ner_file in os.listdir(ner_compressed_data_dir):
        ner_iso = ner_file.replace('.tar.gz', '')
        
        if ner_iso in langs_orig or ner_iso in langs3:
            langrank_supported_train_folders.append(ner_iso)

        if ner_iso in l2v.available_learned_languages() or (ner_iso in LETTER_CODES and LETTER_CODES[ner_iso] in l2v.available_learned_languages()):
            # For the Rahimi splits of WikiANN, all languages that have a test split will also have a dev split
            langrank_supported_dev_folders.append(ner_iso)
            langrank_supported_test_folders.append(ner_iso)


    with open(os.path.join(output_dir, 'langrank_supported_train_folders'), 'w') as f:
        for iso in langrank_supported_train_folders:
            f.write(f'{iso}\n')


    for split in ['dev', 'test']:
        with open(os.path.join(output_dir, f'langrank_supported_{split}_folders'), 'w') as f:
            for iso in langrank_supported_dev_folders:
                f.write(f'{iso}\n')

    
    langrank_train_dscode_to_iso = {f'wiki_en-{iso}' : iso for iso in langrank_supported_train_folders}
    with open(os.path.join(output_dir, 'ner_lang_maps.py'), 'w') as f:
        f.write('LANGRANK_TRAIN_DSCODE_TO_ISO = ' + orjson.dumps(langrank_train_dscode_to_iso, option=orjson.OPT_INDENT_2).decode('utf-8') + '\n')


def create_ner_indexes_2(root_ner_dir='/projects/abeb4417/multilingual_analysis/data/ner', output_dir='/projects/abeb4417/multilingual_analysis/ma_utils/ner_file_iso_maps/'):

    split_min_thresholds = {
        'train' : [('all', 1000), ('medium', 10000), ('large', 15000)],
        'dev' : [('all', 100), ('medium', 1000), ('large', 10000)],
        'test' : [('all', 100), ('medium', 1000), ('large', 10000)],
    }

    allocated_splits = {x : {y : [] for y in ['all', 'medium', 'large']} for x in split_min_thresholds.keys()}

    for split in ['train', 'dev', 'test']:

        split_isos_file = os.path.join(output_dir, f'langrank_supported_{split}_folders')

        with open(split_isos_file, 'r') as f:
            split_isos = [x.strip() for x in f.readlines()]

        print(f'For split {split}, given {len(split_isos)} languages')

        all_num_examples = []
        
        for iso in split_isos:
            data_file = os.path.join(root_ner_dir, iso, f'{split}.txt')

            with open(data_file, 'r') as f:
                num_examples = len(f.readlines())
                all_num_examples.append(num_examples)

            for threshold, threshold_min in split_min_thresholds[split]:
                if num_examples >= threshold_min:
                    allocated_splits[split][threshold].append(iso)

        print(f'Count of all data sizes in split {split}')
        for x in collections.Counter(all_num_examples).most_common():
            print(x)

        for k,v in allocated_splits[split].items():
            print(f'Allocated {len(v)} examples to {k}')


    # Write
    print(f'Allocated splits: {allocated_splits}')
    for split, thresholds in allocated_splits.items():
        for threshold, isos in thresholds.items():
            write_loc = os.path.join(output_dir, f'langrank_{split}_{threshold}_folders')
            with open(write_loc, 'w') as f:
                for iso in isos:
                    f.write(iso + '\n')




if __name__ == '__main__':

    # Create the "all" indices first, which uses the compressed files (so we don't uncompress uncessessary data)
    # create_ner_indexes_1()

    # Then, once the files are copied (see data/scripts/copy_langrank_ner_data.sh) and the number of examples are calculated (see data/scripts/extract_raw_ner_text.sh)
    # Run next method to create "large" and "medium" splits
    # create_ner_indexes_2()


    pass