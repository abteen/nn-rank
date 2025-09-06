import os, sys, argparse, orjson, torch, numpy as np, regex as re, pandas, logging, random

from tqdm import tqdm, trange

from transformers import AutoModelForMaskedLM, AutoTokenizer


"""
    The outputs of this script are of the shape: (num_sentences, seq_len) where each element is the log probability
    
    These log-probabilities can be summed to get the PLL score as defined by (Salazer et al., 2021)
    https://arxiv.org/pdf/1910.14659

    Log probs to PPPL:
            
        token_counts = np.sum(~np.isnan(lang_log_probs))        
        pll = np.nansum(lang_log_probs, axis=1)        
        pppl = np.exp(-1 * np.sum(pll) / token_counts)

"""

def get_inputs(input_ids, vocab_size, max_len):

    x = torch.tensor(input_ids + [tokenizer.pad_token_id] * (args.max_seq_len - len(input_ids)))
    attention_mask = torch.tensor([1] * len(input_ids) + [0] * (args.max_seq_len - len(input_ids)))
    attention_mask = attention_mask.repeat(attention_mask.shape[-1], 1)

    repeats = x.repeat(x.shape[-1], 1)
    mask = torch.ones(x.shape[-1]).diag(0)

    masked_input = repeats.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeats.masked_fill(masked_input != tokenizer.mask_token_id, -100)

    return x, masked_input, labels, attention_mask

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('--shard_idx', type=int)
    parser.add_argument('--num_shards', type=int)

    parser.add_argument('--model_name', type=str)

    parser.add_argument('--experiment_name')
    parser.add_argument('--run_name', nargs='+')

    parser.add_argument('--root_output_dir', default='pppl_outputs/')
    parser.add_argument('--subsample', default=-1, type=int)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=128)

    parser.add_argument('--smoke', action='store_const', const=True, default=False)

    args = parser.parse_args()

    
    # ---- CHECK IF OUTPUT EXISTS ---- #

    input_file_name = os.path.splitext(os.path.basename(args.input))[0]

    run_name = args.run_name[0] if len(args.run_name) == 1 else '/'.join(args.run_name[:-1]) + f'/{os.path.splitext(args.run_name[-1])[0]}'

    output_dir = os.path.join(args.root_output_dir, args.experiment_name, run_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f'{output_dir}/log_probs.npz'

    if os.path.exists(output_file):
        print(f'Found output file (for input {input_file_name}) already -- quitting.')
        quit(0)

    records = read_file(args.input)

    print(f'Read {len(records)} records from: {args.input}')

    records = records[:17] if args.smoke else records

    if args.subsample > 0:

        if args.subsample > len(records):
            print(f'Not subsampling as total records (n={len(records)}) is less than sample size (k={args.subsample})')

        else:

            random.seed(42)
            records = random.sample(records, args.subsample)

            print(f'Subsampled {len(records)} records.')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    for record in tqdm(records):

        enc = tokenizer(
            record['text'],
            max_length=args.max_seq_len,
            truncation=True,
            return_token_type_ids= False,
        )


        tokenized_record = {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'len' : len(enc['input_ids'])
        }

        record['tokenized'] = tokenized_record

    masked_inputs = []
    padded_inputs = []
    labels = []
    attention_masks = []
    lengths = []
    idxs = []

    for record in tqdm(records):

        inpid = record['tokenized']['input_ids']

        padded_input, masked_input, label, attention_mask = get_inputs(inpid, tokenizer.vocab_size, args.max_seq_len)
        padded_inputs.append(padded_input)
        masked_inputs.append(masked_input)
        labels.append(label)
        attention_masks.append(attention_mask)
        lengths.append(len(inpid))

        idxs.append(record['idx'])


    logging.info(f'Attempting to load: {args.model_name}')
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    model = model.to('cuda:0')

    sm = torch.nn.Softmax(dim=2)

    num_batches = args.max_seq_len // args.batch_size

    """
        The pidx (probability_index) tensors are used to lookup the specific logit we are predicting.
        
        The only thing that changes across examples is the specific idx of the vocabulary we are checking.

        Therefore, we only have to calculate this base pidx once.

    """

    pidx_seq_range = torch.arange(0,args.max_seq_len).unsqueeze(1) # Shape (seq_len, 1), vector values: [1 ... seq_len]
    pidx_batch_range = torch.arange(0,args.batch_size).repeat(num_batches).unsqueeze(1) #Shape (seq_len, 1) vector values: [1 ... batch_size] ... [1 ... batch_size]

    pidx_base = torch.hstack((pidx_batch_range, pidx_seq_range)) # Shape (seq_len, 2) vector values: 2 cols

    pidx_base = pidx_base.to('cuda:0')

    probs = []

    for inp, label, attn_mask, length, padded_input in tqdm(zip(masked_inputs, labels, attention_masks, lengths, padded_inputs), total=len(labels)):

        inp = inp.to('cuda:0')
        attn_mask = attn_mask.to('cuda:0')

        # pidx_s indexes the vocabulary (since we are doing masked language modeling the label should be equal to the input)
        pidx_s = padded_input.unsqueeze(1).to('cuda:0') 

        # Horizontally stack this with the base calculation
        # (where_in_batch, where_in_sequence, where_in_vocab)
        # Note: if we were not batching, then where_in_batch == where_in_sequence
        # shape: (seq_len, 3)
        pidx = torch.hstack((pidx_base[:len(pidx_s)], pidx_s))

        inp_batches = inp.split(args.batch_size, dim=0) # Shape (num_batches, batch_size, seq_len)
        pidx_batches = pidx.split(args.batch_size, dim=0) # Shape (num_batches, batch_size, 3)
        am_batches = attn_mask.split(args.batch_size, dim=0)

        batch_probs = []

        for bidx in range(num_batches):
            with torch.inference_mode():
                logits = model(inp_batches[bidx], attention_mask=am_batches[bidx]).logits
                prob = sm(logits) # Shape (batch_size, seq_len, vocab_size)

                taken_probs = prob[pidx_batches[bidx][:,0],pidx_batches[bidx][:,1],pidx_batches[bidx][:,2]] # Shape: (batch_size, 1)
                
                batch_probs.append(torch.log(taken_probs).cpu())

        probs.append(torch.cat(batch_probs)[1:length-1])

    padded_np_array = np.array([np.lib.pad(np.asarray(prob), (0, args.max_seq_len - len(prob)), 'constant', constant_values=np.nan) for prob in probs])

    print(f'Saving to: {output_file}')

    np.savez(output_file, idxs=idxs, log_probs=padded_np_array)

    print('done.')