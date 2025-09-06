import torch, sys
from transformers import AutoTokenizer


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, file, lang, tokenizer):

        self.tokenizer = tokenizer

        self.max_len = 256
        self.assignment = 'last'

        self.create_label2id()
        self.file = file

        self.lang = lang

        self.examples = self.read_file(file)

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

    def create_label2id(self):

        ner_tags = [
            'B-ORG',
            'I-ORG',
            'B-PER',
            'I-PER',
            'B-MISC',
            'I-MISC',
            'B-LOC',
            'I-LOC',
            'O'
        ]

        iter = 0
        self.label2id = {}
        for tag in ner_tags:
            self.label2id[tag] = iter
            iter += 1

        self.label2id['E-ORG'] = self.label2id['I-ORG']
        self.label2id['E-PER'] = self.label2id['I-PER']
        self.label2id['E-MISC'] = self.label2id['I-MISC']
        self.label2id['E-LOC'] = self.label2id['I-LOC']

        for ent in ['ORG', 'PER', 'MISC', 'LOC']:
            for conv in [('E', 'I'), ('S', 'B')]:
                self.label2id[conv[0] + '-' + ent] = self.label2id[conv[1] + '-' + ent]

        self.num_labels = len(set(self.label2id.values()))


    def read_file(self, file, convert_labels=True):

        inps = []
        
        with open(file, 'r') as f:

            temp_tokens = []
            temp_labels = []
            for line in f:
                if line.strip():

                    token = line.strip().split('\t')
                    assert len(token) == 2

                    if convert_labels:
                        temp_tokens.append(token[0].replace(self.lang + ':', ''))
                        temp_labels.append(self.label2id[token[1]])

                    else:
                        temp_tokens.append(token[0].replace(self.lang + ':', ''))
                        temp_labels.append(token[1])

                else:
                    inps.append((temp_tokens,temp_labels))
                    temp_tokens = []
                    temp_labels = []
        return inps

    def encode(self, instance):

        forms = instance[0]
        labels = instance[1]

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
                expanded_labels.append(labels[i])


        s1 = ' '.join(forms)

        enc = self.tokenizer(
            s1,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True,
        )

        if len(expanded_labels) > self.max_len:
            expanded_labels = expanded_labels[:self.max_len]

        enc['labels'] = expanded_labels

        return enc


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')


    with open('ner_error_results.txt', 'w') as f:
        for fpath in sys.argv[1:]:
            iso = fpath.split('/')[-2]
            dataset = NERDataset(fpath,iso, tokenizer)
            f.write(f'{fpath:200s}{dataset.error_rate:.2%}\n')

