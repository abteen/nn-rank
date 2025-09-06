import logging, os, torch, random, sys, time, orjson, argparse
from pprint import pformat

from tqdm import tqdm

from ma_utils.experiment_utils import setup_logging
from ma_utils.metrics import ner_metrics

from ma_utils.custom_datasets.ner_datasets import NERDataset

from transformers import TrainingArguments, BertTokenizer, AutoTokenizer, XLMRobertaTokenizer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer

import traceback

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--experiment_name')
    parser.add_argument('--run_name', nargs='+')
    parser.add_argument('--output_directory')
    parser.add_argument('--log_directory')


    parser.add_argument('--model_to_evaluate')
    parser.add_argument('--base_model')
    parser.add_argument('--eval_files', nargs='+')


    args = parser.parse_args()

    run_name = args.run_name[0] if len(args.run_name) == 1 else '/'.join(args.run_name[:-1]) + f'/{os.path.splitext(args.run_name[-1])[0]}'
    
    setup_logging(args.log_directory, args.experiment_name, run_name)

    output_directory = os.path.join(args.output_directory, args.experiment_name, run_name)
    
    model = args.model_to_evaluate

    logging.info('Loading tokenizer')

    tokenizer_name = args.base_model if os.path.isdir(model) else model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logging.info('Tokenizer: {}'.format(pformat(tokenizer)))

    num_processes = int(os.environ['SLURM_NTASKS'])


    #Load model config and initialize model
    model = AutoModelForTokenClassification.from_pretrained(model)

    #Load Training Arguments
    logging.info('Output directory: {}'.format(output_directory))
    training_arguments = TrainingArguments(output_dir=output_directory,
                                        per_device_eval_batch_size=32)

    #Load collator for MLM
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding='longest'
    )


    #Load Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        data_collator=collator,
        compute_metrics=ner_metrics
    )


    for eval_file in tqdm(args.eval_files):

        logging.info('Current file to evaluate: {}'.format(eval_file))
        eval_iso = eval_file.split('/')[-2]

        eval_dataset = NERDataset(file=eval_file, lang=eval_iso, tokenizer=tokenizer)
        
        results = trainer.predict(eval_dataset)

        print(results)

        data_to_save = {
            'accuracy' : results.metrics['test_accuracy'],
            'f1' : results.metrics['test_f1'],
            'model' : args.model_to_evaluate,
            'tokenizer' : tokenizer_name,
            'eval_file' : eval_file,
            'num_eval_examples' : len(eval_dataset)
        } 

        print(data_to_save)

        logging.info(f"[LOGREADER] {orjson.dumps(data_to_save).decode('utf-8')}")












