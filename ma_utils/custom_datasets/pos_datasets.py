import torch
from conllu import parse_incr
import logging
from transformers import AutoTokenizer
import sys


class ConllPOSDataset(torch.utils.data.Dataset):
    def __init__(self, file, tokenizer, dry_run=False):

        self.tokenizer = tokenizer
        self.max_len = 256
        self.assignment = 'last'
        self.create_label2id()

        if isinstance(file, list):
            logging.info('ConllPOSDataset given list. Will concatenate data from all these files.')
            self.examples = []
            for f in file:
                logging.info(f'Reading file: {f}')
                self.examples.extend(self.read_file(f))
        
        elif isinstance(file, str):
            self.examples = self.read_file(file)
        else:
            raise TypeError(f'Trying to read {file} but is of type {type(file)} which is not supported.')

        if not dry_run:

            self.label_masks = []

            self.filtered_encoded_examples = []
            self.error_encoded_examples = []

            for ex in self.examples:
                encoded_ex = self.encode(ex)
                if len(encoded_ex['input_ids']) == len(encoded_ex['labels']) + 2:
                    self.filtered_encoded_examples.append(encoded_ex)
                else:
                    self.error_encoded_examples.append(encoded_ex)

            print(f'Loaded {len(self.filtered_encoded_examples)} from {len(self.examples)} total examples. Num errors: {len(self.error_encoded_examples)} (rate={len(self.error_encoded_examples)/len(self.examples):.2%})')
            self.error_rate = len(self.error_encoded_examples)/len(self.examples)
            print('----------------------------------------')

    def __getitem__(self, idx):
        return self.filtered_encoded_examples[idx]

    def __len__(self):
        return len(self.filtered_encoded_examples)

    def read_file(self, file, convert_labels=True):

        inps = []
        self.labels_found = []

        with open(file) as f:
            for toklist in parse_incr(f):
                forms = []
                labels = []
                failed_token = False

                for tok in toklist:
                    forms.append(tok['form'])

                    if convert_labels:
                        try:
                            labels.append(self.label2id[tok['upos']])
                        except:
                            failed_token = True
                    else:
                        labels.append(tok['upos'])
                        self.labels_found.append(tok['upos'])

                if not failed_token:
                    inps.append((forms, labels))

        self.labels_found = set(self.labels_found)
        return inps

    def create_label2id(self):

        upos_tags = [
            'ADJ',
            'ADP',
            'PUNCT',
            'ADV',
            'AUX',
            'SYM',
            'INTJ',
            'CCONJ',
            'X',
            'NOUN',
            'DET',
            'PROPN',
            'NUM',
            'VERB',
            'PART',
            'PRON',
            'SCONJ',
            '_'
        ]

        iter = 0
        self.label2id = {}
        for tag in upos_tags:
            self.label2id[tag] = iter
            iter += 1

    def encode(self, instance):

        forms = instance[0]
        labels = instance[1]

        label_mask = []

        expanded_labels = []

        for i in range(0, len(forms)):

            subwords = self.tokenizer.tokenize(forms[i])

            if self.assignment == 'first':
                expanded_labels.append(labels[i])
                for j in range(1, len(subwords)):
                    expanded_labels.append(-100)
            elif self.assignment == 'all':
                for j in range(0,len(subwords)):
                    expanded_labels.append(labels[i])
                    if j < len(subwords) - 1:
                        label_mask.append(0)
                    else:
                        label_mask.append(1)

            elif self.assignment == 'last':
                for j in range(0,len(subwords)-1):
                    expanded_labels.append(-100)
                    label_mask.append(0)
                expanded_labels.append(labels[i])
                label_mask.append(1)

        s1 = ' '.join(forms)

        self.label_masks.append(label_mask)


        enc = self.tokenizer(
            s1,
            max_length=self.max_len,
            truncation=True,
            #padding='max_length',
            return_token_type_ids=True,
            #return_tensors='pt',
        )

        if len(expanded_labels) > self.max_len:
            expanded_labels = expanded_labels[:self.max_len]

        enc['labels'] = expanded_labels


        return enc



if __name__ == '__main__':

    fpath = '/projects/abeb4417/multilingual_analysis/data/pos_ud_data/ud-treebanks-v2.15/UD_Portuguese-PetroGold/pt_petrogold-ud-dev.conllu'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    # dataset = ConllPOSDataset(fpath, tokenizer)

    with open('error_results.txt', 'w') as f:
        for fpath in sys.argv[1:]:
            dataset = ConllPOSDataset(fpath, tokenizer)
            f.write(f'{fpath:200s}{dataset.error_rate:.2%}\n')

