import os, sys, json, collections, random
from tqdm import tqdm

LETTER_CODES = {"am": "amh", "bs": "bos", "vi": "vie", "wa": "wln", "eu": "eus", "so": "som", "el": "ell", "aa": "aar", "or": "ori", "sm": "smo", "gn": "grn", "mi": "mri", "pi": "pli", "ps": "pus", "ms": "msa", "sa": "san", "ko": "kor", "sd": "snd", "hz": "her", "ks": "kas", "fo": "fao", "iu": "iku", "tg": "tgk", "dz": "dzo", "ar": "ara", "fa": "fas", "es": "spa", "my": "mya", "mg": "mlg", "st": "sot", "gu": "guj", "uk": "ukr", "lv": "lav", "to": "ton", "nv": "nav", "kl": "kal", "ka": "kat", "yi": "yid", "pl": "pol", "ht": "hat", "lu": "lub", "fr": "fra", "ia": "ina", "lt": "lit", "om": "orm", "qu": "que", "no": "nor", "sr": "srp", "br": "bre", "rm": "roh", "io": "ido", "gl": "glg", "nb": "nob", "ng": "ndo", "ts": "tso", "nr": "nbl", "ee": "ewe", "bo": "bod", "mt": "mlt", "ta": "tam", "et": "est", "yo": "yor", "tw": "twi", "sl": "slv", "su": "sun", "gv": "glv", "lo": "lao", "af": "afr", "sg": "sag", "sv": "swe", "ne": "nep", "ie": "ile", "bm": "bam", "sc": "srd", "sw": "swa", "nn": "nno", "ho": "hmo", "ak": "aka", "ab": "abk", "ti": "tir", "fy": "fry", "cr": "cre", "sh": "hbs", "ny": "nya", "uz": "uzb", "as": "asm", "ky": "kir", "av": "ava", "ig": "ibo", "zh": "zho", "tr": "tur", "hu": "hun", "pt": "por", "fj": "fij", "hr": "hrv", "it": "ita", "te": "tel", "rw": "kin", "kk": "kaz", "hy": "hye", "wo": "wol", "jv": "jav", "oc": "oci", "kn": "kan", "cu": "chu", "ln": "lin", "ha": "hau", "ru": "rus", "pa": "pan", "cv": "chv", "ss": "ssw", "ki": "kik", "ga": "gle", "dv": "div", "vo": "vol", "lb": "ltz", "ce": "che", "oj": "oji", "th": "tha", "ff": "ful", "kv": "kom", "tk": "tuk", "kr": "kau", "bg": "bul", "tt": "tat", "ml": "mal", "tl": "tgl", "mr": "mar", "hi": "hin", "ku": "kur", "na": "nau", "li": "lim", "nl": "nld", "nd": "nde", "os": "oss", "la": "lat", "bn": "ben", "kw": "cor", "id": "ind", "ay": "aym", "xh": "xho", "zu": "zul", "cs": "ces", "sn": "sna", "de": "deu", "co": "cos", "sk": "slk", "ug": "uig", "rn": "run", "he": "heb", "ba": "bak", "ro": "ron", "be": "bel", "ca": "cat", "kj": "kua", "ja": "jpn", "ch": "cha", "ik": "ipk", "bi": "bis", "an": "arg", "cy": "cym", "tn": "tsn", "mk": "mkd", "ve": "ven", "eo": "epo", "kg": "kon", "km": "khm", "se": "sme", "ii": "iii", "az": "aze", "en": "eng", "ur": "urd", "za": "zha", "is": "isl", "mh": "mah", "mn": "mon", "sq": "sqi", "lg": "lug", "gd": "gla", "fi": "fin", "ty": "tah", "da": "dan", "si": "sin", "ae": "ave", "alb": "sqi", "arm": "hye", "baq": "eus", "tib": "bod", "bur": "mya", "cze": "ces", "chi": "zho", "wel": "cym", "ger": "deu", "dut": "nld", "gre": "ell", "per": "fas", "fre": "fra", "geo": "kat", "ice": "isl", "mac": "mkd", "mao": "mri", "may": "msa", "rum": "ron", "slo": "slk"}

"""
Outputs file maps in column format:

[task folder] [task file name] [task iso] [bible iso] [bible file]


"""


def create_pos_bible_indices(bible_dir='/projects/abeb4417/old/lrl/data/bibles_raw/', task_maps='pos_file_iso_maps/', output_dir='bible_file_iso_maps/'):

    bible_data = collections.defaultdict(list)
    available_bible_files = os.listdir(bible_dir)
    for bible_file in tqdm(available_bible_files):
        p = os.path.join(bible_dir, bible_file)
        # with open(p, 'r') as f:
        #     valid_lines = [x for x in f.readlines() if x.strip()]
        num_valid_lines = random.randint(0,40000) # :)
        
        bible_iso = bible_file.split('-')[0]
        bible_data[bible_iso].append((bible_file, num_valid_lines))

    for k in bible_data.keys():
        v = bible_data[k]
        v.sort(key = lambda x : x[1], reverse=True)
        bible_data[k] = v
    
    
    available_bible_isos = set(bible_data.keys())
    splits = ['train', 'dev', 'test']
    sizes = ['large', 'medium', 'all']


    pos_output_dir = os.path.join(output_dir, 'pos/')
    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)

    pos_split_to_bible = {}
    for split in splits:
        for size in sizes:
            full_split = f'{split}_{size}'

            data_to_write = []

            with open(os.path.join(task_maps, f'langrank_{full_split}_fnames'), 'r') as split_files, open(os.path.join(task_maps, f'langrank_{full_split}_folders'), 'r') as split_folders:

                pos_files = [x.strip() for x in split_files.readlines()]
                pos_folders = [x.strip() for x in split_folders.readlines()]

                assert len(pos_files) == len(pos_folders)

            pos_isos = [x.split('_')[0] for x in pos_files]
            
            for i, piso in enumerate(pos_isos):
                three_digit_iso = LETTER_CODES.get(piso, None)
                if three_digit_iso in available_bible_isos: # or :

                    temp_data = [
                        pos_folders[i],
                        pos_files[i],
                        piso,
                        three_digit_iso,
                        bible_data[three_digit_iso][0][0]
                    ]

                    data_to_write.append(temp_data)
            write_dir = os.path.join(pos_output_dir, f'langrank_bible_{full_split}_columns')
            with open(write_dir, 'w') as f:
                for d in data_to_write:
                    f.write('\t'.join(d))
                    f.write('\n')

def create_ner_bible_indices(bible_dir='/projects/abeb4417/old/lrl/data/bibles_raw/', task_maps='ner_file_iso_maps/', output_dir='bible_file_iso_maps/'):

    bible_data = collections.defaultdict(list)
    available_bible_files = os.listdir(bible_dir)
    for bible_file in tqdm(available_bible_files):
        p = os.path.join(bible_dir, bible_file)
        # with open(p, 'r') as f:
        #     valid_lines = [x for x in f.readlines() if x.strip()]
        num_valid_lines = random.randint(0,40000) # :)
        
        bible_iso = bible_file.split('-')[0]
        bible_data[bible_iso].append((bible_file, num_valid_lines))

    for k in bible_data.keys():
        v = bible_data[k]
        v.sort(key = lambda x : x[1], reverse=True)
        bible_data[k] = v
    
    
    available_bible_isos = set(bible_data.keys())
    splits = ['train', 'dev', 'test']
    sizes = ['large', 'medium', 'all']


    ner_output_dir = os.path.join(output_dir, 'ner/')
    if not os.path.exists(ner_output_dir):
        os.makedirs(ner_output_dir)

    ner_split_to_bible = {}
    for split in splits:
        for size in sizes:
            full_split = f'{split}_{size}'

            data_to_write = []

            with open(os.path.join(task_maps, f'langrank_{full_split}_folders'), 'r') as split_folders:

                ner_folders = [x.strip() for x in split_folders.readlines()]
            
            for i, niso in enumerate(ner_folders): #ner folders are just their iso
                three_digit_iso = LETTER_CODES.get(niso, None)
                
                if niso in available_bible_isos: # or :
                    
                    temp_data = [
                        ner_folders[i],
                        ner_folders[i],
                        niso,
                        niso,
                        bible_data[niso][0][0]
                    ]

                    data_to_write.append(temp_data)

                elif three_digit_iso in available_bible_isos: # or :

                    temp_data = [
                        ner_folders[i],
                        ner_folders[i],
                        niso,
                        three_digit_iso,
                        bible_data[three_digit_iso][0][0]
                    ]

                    data_to_write.append(temp_data)

            write_dir = os.path.join(ner_output_dir, f'langrank_bible_{full_split}_columns')
            with open(write_dir, 'w') as f:
                for d in data_to_write:
                    f.write('\t'.join(d))
                    f.write('\n')

if __name__ == '__main__':

    create_pos_bible_indices()
    create_ner_bible_indices()