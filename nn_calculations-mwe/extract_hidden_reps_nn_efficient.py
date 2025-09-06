import argparse
import numpy as np
import torch
import os
from tqdm import tqdm, trange
import logging
from datetime import datetime
import orjson

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import logging as trf_logging

def read_file(fpath):

    records = []

    with open(fpath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                records.append({
                    'idx' : i,
                    'text' : line
                })

    return records

def setup_logging(log_dir, experiment_name, run_name):
    
    intended_log_dir = os.path.join(log_dir, experiment_name, run_name)

    date_time = datetime.now().strftime("%Y%d%m-%H%M%S")

    if not os.path.isdir(intended_log_dir):
        os.makedirs(intended_log_dir)

    # Setup logging
    logging.root.handlers = []
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}/run-{}.log".format(intended_log_dir, date_time)),
            logging.StreamHandler()
        ]
    )

    trf_logging.set_verbosity_info()
    trf_logging.enable_propagation()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_name')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--experiment_name', default='')
    parser.add_argument('--run_name_extractor', help='Define how to set the output name for each input file. Choose from: [ud, ner]')
    parser.add_argument('--run_name', nargs='+', type=str)
    parser.add_argument('--root_output_dir', default='hidden_rep_outputs/')
    parser.add_argument('--max_seq_len', default=512)
    parser.add_argument('--limiter', default=None, type=int)
    parser.add_argument('--layer', default=-1, type=int, help='Only extract data from a specific layer. Helpful if not doing ablations -- saves memory')
    parser.add_argument('--max_num_examples_before_stream_save', default=25000, type=int)
    parser.add_argument('--save_every', default=10000, type=int)
    parser.add_argument('--log_dir', default='logs/extract_hr_efficient_logs/')

    args = parser.parse_args()

    # ---- LOAD TOKENIZER ---- #

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- LOAD MODEL ---- #

    config = AutoConfig.from_pretrained(args.model_name, output_hidden_states=True, output_attentions=True)
    model = AutoModel.from_pretrained(args.model_name, config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    model = model.to(device)

    logging.info(f'Iterating over inputs to remove duplicates')
    seen_inputs = []
    for inp in args.inputs:
        if inp not in seen_inputs:
            seen_inputs.append(inp)
            logging.info(f'Keeping: {inp}')
        else:
            logging.info(f'Skipping: {inp}')

    logging.info(f'Reduced length of inputs from {len(args.inputs)} to {len(seen_inputs)}')

    for input_file in tqdm(seen_inputs):

        # ---- SETUP SAVE DIRS ---- # 

        if args.run_name_extractor == 'ud':
            current_run_name = args.run_name + [os.path.basename(input_file)]
        elif args.run_name_extractor == 'ner':
            current_run_name = args.run_name + [input_file.split('/')[-2]]
        elif args.run_name_extractor == 'bible':
            current_run_name = args.run_name + [os.path.splitext(os.path.basename(input_file))[0]]
        else:
            raise ValueError(f'args.run_name_extractor set to "{args.run_name_extractor}" where only ["ud", "ner", "bible"] are supported. ')

        run_name = current_run_name[0] if len(current_run_name) == 1 else '/'.join([os.path.splitext(rn)[0] for rn in current_run_name])
       

        output_dir = os.path.join(args.root_output_dir, args.experiment_name, run_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # ---- LOGGING ---- #

        setup_logging(args.log_dir, args.experiment_name, run_name)

        logging.info(f'Setting output directory to: {output_dir}/')
        logging.info(f'HERE: {input_file}')

        # ---- LOAD DATA ---- #

        if os.path.exists(input_file):
            records = read_file(input_file)
        else:
            logging.error(f'could not find file. quitting.')

        logging.info(f'Read {len(records)} records from: {input_file}')

        records = records[:args.limiter] if args.limiter else records

        stream_save = False

        if len(records) > args.max_num_examples_before_stream_save:
            stream_save = True
            logging.info(f'Length of records is longer than max_num_examples_before_stream_save. Will save examples every {args.save_every} iterations.')
        
        # ---- TOKENIZE DATA ---- #


        for record in tqdm(records):

            enc = tokenizer(
                record['text'],
                max_length=args.max_seq_len,
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
        attention_outputs = []
        input_id_outputs = []
        idxs = []

        layer_string = f"layer_{args.layer}" if args.layer >= 0 else "layer_all"

        with torch.inference_mode():

            for i, record in enumerate(tqdm(records)):

                input_ids = record['tokenized']['input_ids'].to(device)
                attn_mask = record['tokenized']['attention_mask'].to(device)

                output = model(input_ids = input_ids, attention_mask = attn_mask)

                # Hidden states is a tuple where each index corresponds to the layer, index 0 is embedding
                # Each value of a tuple is a tensor of size (batch_size, seq_len, hidden_dim)
                hidden_states = output['hidden_states']
                hidden_states = torch.concat(hidden_states, axis=0)

                # attentions is a tuple where each index corresponds to the layer, index 0 is layer 1
                # each value of the tuple is a tensor of size (batch_size, num_heads, seq_len, seq_len)
                attentions = output['attentions']
                attentions = torch.concat(attentions, axis=0)

                if args.layer >= 0:
                    hidden_states = hidden_states[args.layer]
                    attentions = attentions[args.layer]

                hidden_state_outputs.append(hidden_states.detach().cpu())
                attention_outputs.append(attentions.detach().cpu())
                input_id_outputs.append(input_ids.detach().cpu())
                idxs.append(record['idx'])

                if stream_save and i % args.save_every == 0:
                    logging.info(f'Reached iteration {i}, saving outputs.')

                    for i in trange(len(hidden_state_outputs)):
                        np.save(file=os.path.join(output_dir, f'{idxs[i]}_hidden_{layer_string}.npy'), arr = hidden_state_outputs[i].numpy())
                        np.save(file=os.path.join(output_dir, f'{idxs[i]}_attentions_{layer_string}.npy'), arr = attention_outputs[i].numpy())
                        np.save(file=os.path.join(output_dir, f'{idxs[i]}_input_ids_{layer_string}.npy'), arr = input_id_outputs[i].numpy())

                    hidden_state_outputs = []
                    attention_outputs = []
                    input_id_outputs = []
                    idxs = []
        
        logging.info(f'Finished processing. Number of unsaved outputs after iterations: {len(hidden_state_outputs)}')

        idxs = torch.tensor(idxs)

        assert len(hidden_state_outputs) == len(idxs)

        for i in trange(len(hidden_state_outputs)):
                        np.save(file=os.path.join(output_dir, f'{idxs[i]}_hidden_{layer_string}.npy'), arr = hidden_state_outputs[i].numpy())
                        np.save(file=os.path.join(output_dir, f'{idxs[i]}_attentions_{layer_string}.npy'), arr = attention_outputs[i].numpy())
                        np.save(file=os.path.join(output_dir, f'{idxs[i]}_input_ids_{layer_string}.npy'), arr = input_id_outputs[i].numpy())

        experiment_metadata = vars(args)
        experiment_metadata['save_datetime'] = datetime.now().strftime("%Y%d%m-%H%M%S")
        experiment_metadata['input_file'] = input_file

        with open(os.path.join(output_dir, 'metadata.jsonl'), 'wb') as f:
            f.write(orjson.dumps(experiment_metadata))
        logging.info('done.')

    



