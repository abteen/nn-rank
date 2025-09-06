import os, argparse, orjson, pandas
from datetime import datetime
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('log_dir_to_read')
    parser.add_argument('--save_dir', default='processed_eval_outputs/')
    parser.add_argument('--save_fname', required=True)
    parser.add_argument('--eval_split')
    parser.add_argument('--source_split')

    args = parser.parse_args()


    logs = os.listdir(args.log_dir_to_read)

    records = []

    num_per_log = 0

    for log in tqdm(logs):

        fpath = os.path.join(args.log_dir_to_read, log)

        with open(fpath, 'r') as f:
            for line in f:
                if '[LOGREADER]' in line:
                    d = line.strip().split("[LOGREADER] ")[1]
                    data = orjson.loads(d)
                    records.append(data)
                    num_per_log += 1
        print(f'{log} {num_per_log}')
        num_per_log = 0

    df = pandas.DataFrame.from_records(records)

    print(df['f1'].unique())
    print(df['eval_file'].tolist()[0])

    df['train_file'] = df['model'].apply(lambda x :os.path.basename(x.replace('/final_model', '') + f'_{args.source_split}'))
    df['eval_file'] =  df['eval_file'].apply(lambda x : x.split('/')[-2] + f'_{args.eval_split}')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    df.to_csv(os.path.join(args.save_dir, args.save_fname))

    metadata = {
        'load_dir' : args.log_dir_to_read,
        'num_files_loaded' : len(os.listdir(args.log_dir_to_read)),
        'time_of_saving' : datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    }

    with open(os.path.join(args.save_dir, 'metadata.jsonl'), 'wb') as f:
        f.write(orjson.dumps(metadata))

