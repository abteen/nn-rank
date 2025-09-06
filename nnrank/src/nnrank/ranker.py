import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
from tqdm import tqdm, trange
from collections import Counter

class NNRanker():
    """Rank source languages for cross-lingual transfer.
    
    For both source and target languages, we hold information in memory using the following numpy arrays:

        target_indices: (N_tgt,)              # Map each target token to the line index from the corpus
        target_input_ids: (N_tgt,)            # Map each target token to its input id given to the model
        T: (N_tgt, 768)                       # Map each target token to it's hidden representation
        target_langs: (N_tgt,)                # Map each target token to the language of the corpus it was taken from.

        all_source_indices: (N_src,)          # Map every source token to the line index from the corpus
        all_source_input_ids: (N_src,)        # Map every source token to its input id given to the model
        S: (N_src, 768)                       # Map every source token to it's hidden representation
        all_source_langs: (N_src,)            # Map every source token to the language of the corpus it was taken from.

    To construct S, we concatenate the hidden_reps array along axis 0. T is just the hidden_reps array for the target language.
    On the source language side, we want to concatenate all other arrays as well.

    When calculating neighbors with FAISS, for each target token it returns k neighbor indexes. We can use these to lookup the source
    language from the concatenated langs array, input_ids from the concatenated input_ids array, etc.:

        self.index = faiss.IndexFlatIP(S.shape[1])
        self.index.add(S)

        distances, neighbors = self.index.search(T, k=k)

        neighbors[0] : neighbors of first token of first loaded line
        all_source_langs[neighbors[0]] : language/unique identifier of the corpus from which the neighboring tokens are from


    When writing to disk, we save the input_ids and hidden reps in one file per line. 

    Attributes:
        model: A huggingface model identifier. Models saved locally could also be used as long as the config is set correctly.
        layer: The layer from which to extract hidden representations.
        config: Huggingface config, needed for hidden dimension and max sequence length
        data: Dictionary holding loaded text
        hidden_representations: Dictionary holding extracted hidden representations and input ids
        source_languages: list of source languages
        index: FAISS index 
    """
    def __init__(self, model, layer=8):
        self.model = model
        self.layer = layer
        self.config = AutoConfig.from_pretrained(model, output_hidden_states=True, output_attentions=True)

        self.data = {}
        self.hidden_representations = {}

        self.source_languages = []

        self.index = None

    def load_file(self, fpath, limiter):
        """Load text from a file.

        Args:
            fpath (str): file path to load
            limiter (int): limit loading to first 

        Returns:
            list: list of dictionaries
        """
        records = []

        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    records.append({
                        'idx' : i,
                        'text' : line
                    })

                if len(records) >= limiter:
                    break

        return records

    def load_text_data(self, language_list, limiter):
        """Load all data.

        Args:
            language_list (dictionary): map unique identifier to string associated with file path or list of already loaded text
            limiter (int): limit loading to first k files

        Raises:
            TypeError: If language_list values are not strings or list of text.
        """
        if isinstance(language_list, dict):

                for lang_name, lang_data in language_list.items():

                    if isinstance(lang_data, str) and os.path.isfile(lang_data):
                        self.data[lang_name] = self.load_file(lang_data, limiter)
                        print(f'{lang_name:<50s} Loaded {len(self.data[lang_name])} lines from {lang_data}')
                    
                    elif isinstance(lang_data, list):
                        self.data[lang_name] = [{'idx' : i, 'text' : text} for i, text in enumerate(lang_data)]
                        print(f'{lang_name:<50s} Loaded {len(self.data[lang_name])} lines from memory')

                    else:
                        raise TypeError(f'language_list values should be file path or list of loaded data, not {lang_data}')

    def set_source_languages(self, source_languages):
        """Set source languages.

        Args:
            source_languages (list): list of source languages denoted by unique key
        """
        self.source_languages = []
        for sl in source_languages:
            if sl not in self.data:
                print(f'Source language key: {{{sl}}} not in loaded data keys.')
            self.source_languages.append(sl)

        print(f'Number of source languages {len(self.source_languages)}.')

    def get_source_languages(self):
        return self.source_languages

    def extract_hidden_representations(self, 
                                      keys=None,
                                      root_output_dir='hidden_representations/',
                                      strip_first_last=True,
                                      keep_in_memory=True,
                                      stream_save=False,
                                      save_to_disk=False,
                                      save_every=1000,
                                      max_seq_len=512):
        """Extract hidden representations from the model.

        Args:
            keys (list, optional): List of keys to extract hidden representations for. Helpful for preventing duplicate extractoin Defaults to None.
            root_output_dir (str, optional): Directory to save hidden representations to, if stream_save or save_to_disk set to True. Defaults to 'hidden_representations/'.
            strip_first_last (bool, optional): Strip first and last input id/hidden rep. Used to remove special tokens. Defaults to True.
            keep_in_memory (bool, optional): Keep extracted model data in memory. Defaults to True.
            stream_save (bool, optional): Save extracted data to disk during the process. Defaults to False.
            save_to_disk (bool, optional): Save extracted data to disk after processing. Defaults to False.
            save_every (int, optional): Number of iterations before stream_save. Defaults to 1000.
            max_seq_len (int, optional): Maximum sequence length for truncation. Defaults to 512.
        """
        keys = keys if keys is not None else list(self.data.keys())

        model = AutoModel.from_pretrained(self.model, config=self.config)
        tokenizer = AutoTokenizer.from_pretrained(self.model)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'using device: {device}')
        model = model.to(device)

        for key in keys:

            if key not in self.data:
                print(f'Key {key} not in loaded data. Skipping.')
                continue

            if stream_save or save_to_disk:
                output_dir = os.path.join(root_output_dir, key)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

            records = self.data[key]

            for record in tqdm(records):

                enc = tokenizer(
                    record['text'],
                    max_length=max_seq_len,
                    truncation=True,
                    return_token_type_ids= False,
                    return_tensors='pt'
                )

                tokenized_record = {
                    'input_ids': enc['input_ids'],
                    'attention_mask': enc['attention_mask'],
                    'len' : len(enc['input_ids'][0])
                }

                record['tokenized'] = tokenized_record

            hidden_state_outputs = []
            input_id_outputs = []
            idxs = []
            idxs_per_hidden_state = []

            layer_string = f"layer_{self.layer}"

            with torch.inference_mode():

                for i, record in enumerate(tqdm(records)):

                    input_ids = record['tokenized']['input_ids'].to(device)
                    attn_mask = record['tokenized']['attention_mask'].to(device)

                    output = model(input_ids = input_ids, attention_mask = attn_mask)

                    # Hidden states is a tuple where each index corresponds to the layer, index 0 is embedding
                    # Each value of a tuple is a tensor of size (batch_size, seq_len, hidden_dim)
                    hidden_states = output['hidden_states']
                    hidden_states = torch.concat(hidden_states, axis=0)

                    hidden_states = hidden_states[self.layer]
                    input_ids = input_ids.squeeze(axis=0)

                    if strip_first_last:
                        hidden_states = hidden_states[1:-1]
                        input_ids = input_ids[1:-1]

                    hidden_state_outputs.append(hidden_states.detach().cpu())
                    input_id_outputs.append(input_ids.detach().cpu())
                    idxs.append(record['idx'])
                    idxs_per_hidden_state.extend([record['idx']] * len(input_ids))

                    if stream_save and i % save_every == 0:
                        print(f'Reached iteration {i}, saving outputs.')

                        for i in trange(len(hidden_state_outputs)):
                            np.save(file=os.path.join(output_dir, f'{idxs[i]}_hidden_{layer_string}.npy'),
                                    arr = hidden_state_outputs[i].numpy())
                            np.save(file=os.path.join(output_dir, f'{idxs[i]}_input_ids_{layer_string}.npy'),
                                    arr = input_id_outputs[i].numpy())

                        hidden_state_outputs = []
                        input_id_outputs = []
                        idxs = []
            
            print(f'Finished processing {key}.')

            idxs = torch.tensor(idxs)
            idxs_per_hidden_state = torch.tensor(idxs_per_hidden_state).numpy()

            assert len(hidden_state_outputs) == len(idxs)

            if stream_save or save_to_disk:
                for i in trange(len(hidden_state_outputs)):
                                np.save(file=os.path.join(output_dir, f'{idxs[i]}_hidden_{layer_string}.npy'),
                                        arr = hidden_state_outputs[i].numpy())
                                np.save(file=os.path.join(output_dir, f'{idxs[i]}_input_ids_{layer_string}.npy'),
                                        arr = input_id_outputs[i].numpy())

            if keep_in_memory:
                
                if stream_save:
                    print(f'Cannot set both stream_save and keep_in_memory to True.')
                    return

                self.hidden_representations[key] = {
                    'hidden_state_outputs' : np.concatenate(hidden_state_outputs, axis=0),
                    'input_id_outputs' : np.concatenate(input_id_outputs, axis=0),
                    'idxs' : idxs_per_hidden_state
                }

    def save_hidden_representations(self, keys=None, root_output_dir='hidden_representations/'):
        """Save hidden representations to disk.

        Args:
            keys (list, optional): List of keys we want to save representations of. Defaults to None.
            root_output_dir (str, optional): Directory to save in. Defaults to 'hidden_representations/'.
        """
        keys = keys if keys is not None else list(self.hidden_representations.keys())

        layer_string = f'layer_{self.layer}'

        for key in keys:
            if key not in self.hidden_representations:
                print(f'Key {key} not in loaded data. Skipping.')
                continue

            hidden_state_outputs = self.hidden_representations[key]['hidden_state_outputs']
            input_id_outputs = self.hidden_representations[key]['input_id_outputs']
            idxs = self.hidden_representations[key]['idxs']

            output_dir = os.path.join(root_output_dir, key)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            unique_idxs = np.unique(idxs)

            for idx in unique_idxs:
                indexer = np.where(idxs == idx)

                hs = hidden_state_outputs[indexer]
                iids = input_id_outputs[indexer]

                np.save(file=os.path.join(output_dir, f'{idx}_hidden_{layer_string}.npy'), arr = hs)
                np.save(file=os.path.join(output_dir, f'{idx}_input_ids_{layer_string}.npy'), arr = iids)

    def load_hidden_representations(self, keys, root_load_dir):
        """Load hidden representations from disk.

        Args:
            keys (list): Unique keys to load.
            root_load_dir (str): Directory to load from.
        """
        for key in keys:

            load_dir = os.path.join(root_load_dir, key)

            layer_string = f'layer_{self.layer}'

            all_files = [f for f in os.listdir(load_dir) if '.npy' in f and layer_string in f]
            found_indices = list(set([int(x.split('_')[0]) for x in all_files]))
            found_indices.sort()

            all_indices = np.empty(0, dtype=int)
            all_input_ids = np.empty(0, dtype=int)
            all_hidden_reps = np.empty((0,self.config.hidden_size))

            for idx in tqdm(found_indices):

                hidden_file = os.path.join(load_dir, f'{idx}_hidden_{layer_string}.npy')
                input_ids_file = os.path.join(load_dir, f'{idx}_input_ids_{layer_string}.npy')

                input_ids = np.load(input_ids_file)
                all_input_ids = np.concatenate([all_input_ids, input_ids], axis=0)
                
                hidden_reps = np.load(hidden_file)
                all_hidden_reps = np.concatenate([all_hidden_reps, hidden_reps], axis=0)

                indices = np.repeat(idx, len(input_ids))
                all_indices = np.concatenate([all_indices, indices], axis=0)

            print(f'Loaded {len(all_hidden_reps)} hidden representations from {len(found_indices)} input indices in dir {load_dir}.')

            self.hidden_representations[key] = {
                    'hidden_state_outputs' : all_hidden_reps,
                    'input_id_outputs' : all_input_ids,
                    'idxs' : all_indices
                }

    def rank(self, keys, k=5, recalculate_source=False):
        """Rank source languages for each target language.

        Args:
            k (int, optional): Number of neighbors in calculation. Defaults to 5.
            recalculate_source (bool, optional): Force re-indexing of FAISS index. Defaults to False.

        Returns:
            dict: dictionary mapping each target language to its ranking.
        """
        print(f'Generating ranking with {len(self.source_languages)} source languages and {len(keys)} target languages')

        if recalculate_source or self.index is None:
            print(f'Creating FAISS index...')
            S = np.concatenate([self.hidden_representations[sl]['hidden_state_outputs'] for sl in self.source_languages])
            self.all_source_langs = np.concatenate([np.repeat(sl, len(self.hidden_representations[sl]['input_id_outputs'])) for sl in self.source_languages])    

            self.index = faiss.IndexFlatIP(S.shape[1])
            self.index.add(S)
        else:
            print(f'FAISS index already calculated')

        ranking = {}

        for tl in keys:

            tl_result = Counter({sl : 0 for sl in self.source_languages})

            T = self.hidden_representations[tl]['hidden_state_outputs']
            distances, neighbors = self.index.search(T, k=k)

            for nidx in range(neighbors.shape[0]):
                token_neighbors = neighbors[nidx]
                tl_result.update(self.all_source_langs[token_neighbors])
                
            ranking[tl] = sorted([(lang, count) for lang, count in tl_result.items()], key = lambda x : x[1], reverse=True)

        self.ranking = ranking
        return ranking
            
    def verbose_ranking(self, keys, k=5):
        """Calculate ranking and show surface forms of tokens.

        Args:
            k (int, optional): Number of nearest neighbors. Defaults to 5.

        Returns:
            dict: dictionary mapping from target language to ranking
        """
        print(f'Generating ranking with {len(self.source_languages)} source languages and {len(keys)} target languages')

        # Create S matrix
        all_source_indices = np.concatenate([self.hidden_representations[sl]['idxs'] for sl in self.source_languages])
        all_source_input_ids = np.concatenate([self.hidden_representations[sl]['input_id_outputs'] for sl in self.source_languages])
        S = np.concatenate([self.hidden_representations[sl]['hidden_state_outputs'] for sl in self.source_languages])
        all_source_langs = np.concatenate([np.repeat(sl, len(self.hidden_representations[sl]['input_id_outputs'])) for sl in self.source_languages])    

        index = faiss.IndexFlatIP(S.shape[1])
        index.add(S)

        counts = {}
        ranking = {}

        tokenizer = AutoTokenizer.from_pretrained(self.model)

        for tl in keys:

            if tl not in self.hidden_representations:
                print(f'target language {tl} not loaded. skipping')
                continue

            tl_result = Counter({sl : 0 for sl in self.source_languages})

            target_indices = self.hidden_representations[tl]['idxs']
            target_input_ids = self.hidden_representations[tl]['input_id_outputs']
            T = self.hidden_representations[tl]['hidden_state_outputs']
        
            distances, neighbors = index.search(T, k=k)

            for tidx in np.unique(target_indices):
                print('*' * 50)
                print(f'(target lang = {tl}) Index: {tidx}')
                indexer = np.where(target_indices == tidx)[0]
                print(f'\tSurface form tokens: {tokenizer.convert_ids_to_tokens(target_input_ids[indexer])}')
                print(f'\tInput Ids: {list(target_input_ids[indexer])}')
                for iidx in indexer:
                    print(f'Target Token: {tokenizer.convert_ids_to_tokens([target_input_ids[iidx]])}')
                    for i, n in enumerate(neighbors[iidx]):
                        print(f'\tNeighbor {i+1}: {tokenizer.convert_ids_to_tokens([all_source_input_ids[n]])!s:>20} (lang={all_source_langs[n]})')
                
                break
                
            counts[tl] = tl_result
            ranking[tl] = sorted([(lang, count) for lang, count in tl_result.items()], key = lambda x : x[1], reverse=True)

        self.ranking = ranking
        return ranking


