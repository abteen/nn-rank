import os, sys, argparse, random, numpy as np, pandas
from transformers import AutoTokenizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_files', nargs='+')
    parser.add_argument('--model_name')
    parser.add_argument('--experiment_name')
    parser.add_argument('--root_output_dir', default='outputs/')
    parser.add_argument('--run_name', nargs='+')
    parser.add_argument('--subsample', type=int)

    args = parser.parse_args()

    run_name = args.run_name[0] if len(args.run_name) == 1 else '/'.join(args.run_name[:-1]) + f'/{os.path.splitext(args.run_name[-1])[0]}'

    output_dir = os.path.join(args.root_output_dir, args.experiment_name, run_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f'{output_dir}/unk_counts.tsv'

    # Load Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    unk_id = tokenizer.unk_token_id
    pad_id = tokenizer.pad_token_id

    print(f'Setting unk id to: {unk_id}')
    print(f'Setting pad_id to: {pad_id}')

    records = []

    for input_file in args.input_files:

        # Read input
        with open(input_file, 'r') as f:
            lines = [x.strip() for x in f.readlines()]

        if args.subsample:
            random.seed(42)
            lines = random.choices(lines, k=min(args.subsample, len(lines)))

            print(f'Subsampling input to length: {len(lines)}')
        else:
            print(f'Loaded {len(lines)} lines')


        # Tokenize
        tokens = tokenizer(lines, return_tensors='np', add_special_tokens=False, padding='longest')['input_ids']

        total_non_pad_tokens = np.count_nonzero(tokens != pad_id)
        total_unk_tokens = np.count_nonzero(tokens == unk_id)
        
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]


        records.append({
            'input_file': input_file_name,
            'tokenizer' : args.model_name,
            'total_unk_tokens' : total_unk_tokens,
            'total_non_pad_tokens' : total_non_pad_tokens
        })

    df = pandas.DataFrame.from_records(records)
    df = df.sort_values(by='input_file')
    df.to_csv(output_file, index=False, sep='\t')