import logging, os, torch, random, sys, time, argparse
from omegaconf import OmegaConf

from ma_utils.experiment_utils import set_seeds, setup_logging, check_experiment_setup
from ma_utils.metrics import ner_metrics

from ma_utils.custom_datasets.ner_datasets import NERDataset

from datetime import datetime

from transformers import TrainingArguments, BertTokenizer, AutoTokenizer, XLMRobertaTokenizer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--training_config')

    parser.add_argument('--experiment_name')
    parser.add_argument('--run_name', nargs='+')
    parser.add_argument('--output_directory')
    parser.add_argument('--log_directory')


    parser.add_argument('--model')
    parser.add_argument('--train_iso')
    parser.add_argument('--train_file')
    parser.add_argument('--dev_file')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    run_name = args.run_name[0] if len(args.run_name) == 1 else '/'.join(args.run_name[:-1]) + f'/{os.path.splitext(args.run_name[-1])[0]}'
    
    date_time = datetime.now().strftime("%Y%d%m-%H%M%S")
    log_fname = os.path.splitext(os.path.basename(args.train_file))[0]
    setup_logging(args.log_directory, args.experiment_name, run_name, log_file_name=f'{log_fname}-{date_time}')
    
    set_seeds(args.seed)
    output_directory = os.path.join(args.output_directory, args.experiment_name, run_name)

    train_config = OmegaConf.merge(OmegaConf.from_cli(), OmegaConf.load(args.training_config))

    check_experiment_setup(train_config, output_directory)

    model = args.model
    tokenizer_dict = {
        'bert-base-multilingual-cased' : BertTokenizer,
        'xlm-roberta-base' : XLMRobertaTokenizer
    }

    logging.info('Loading tokenizer')
    tok_model = tokenizer_dict[model]
    tokenizer = tok_model.from_pretrained(model)

    num_processes = int(os.environ['SLURM_NTASKS'])

    train_file_to_load = args.train_file

    if train_file_to_load is None:
        logging.error('train file did not load, skipping')
        quit()


    train_dataset = NERDataset(file=train_file_to_load, lang=args.train_iso, tokenizer=tokenizer)

    logging.info(f'loaded train file from {train_file_to_load}')

    #Load model config and initialize model
    model = AutoModelForTokenClassification.from_pretrained(model, num_labels=train_dataset.num_labels)

    logging.info('Final loaded training data: {}'.format(train_dataset))

    #Load Training Arguments
    logging.info('Output directory: {}'.format(output_directory))
    training_arguments = TrainingArguments(output_dir=output_directory,
                                        **train_config['training_arguments'])

    #Load collator for MLM
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding='longest'
    )


    #Load Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        data_collator=collator,
        compute_metrics=ner_metrics
    )

    trainer.train()


    model.save_pretrained(os.path.join(output_directory, 'final_model'))
    logging.info('Model saved in: {}'.format(output_directory))
    logging.info('-'*25 + 'Finished with train input: {}'.format(train_file_to_load) + '-'*25)

        














