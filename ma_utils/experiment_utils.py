import argparse, logging, git, os, sys
import numpy, torch
from transformers import set_seed, logging as trf_logging
from omegaconf import OmegaConf
from datetime import datetime


def set_seeds(seed=42):
    set_seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_datetime_str():
    return datetime.now().strftime("%Y%d%m-%H%M%S")

def setup_logging(log_directory, experiment_name, run_name, log_file_name=None):
    
    intended_log_dir = os.path.join(log_directory, experiment_name, run_name)

    date_time = datetime.now().strftime("%Y%d%m-%H%M%S")

    fname = f'{log_file_name}-{date_time}' if log_file_name is not None else date_time

    if not os.path.isdir(intended_log_dir):
        os.makedirs(intended_log_dir)

    # Setup logging
    logging.root.handlers = []
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}/{}.log".format(intended_log_dir, fname)),
            logging.StreamHandler()
        ]
    )

    trf_logging.set_verbosity_info()
    trf_logging.enable_propagation()


def check_experiment_setup(config, output_directory):


    if config.check_already_trained:
        final_model_directory = os.path.join(output_directory, 'final_model')
        if os.path.isdir(final_model_directory) and 'pytorch_model.bin' in os.listdir(final_model_directory):
            logging.info('Final model for this experiment exists. Exitting.')
            sys.exit(0)


    if config.check_git_status:
        repo = git.Repo(search_parent_directories=True)
        commit_id = repo.head.object.hexsha
        branch = repo.active_branch.name
        logging.info('Using code from git commit: {}'.format(commit_id))
        logging.info('On active branch: {}'.format(branch))

    # Set visible devices

    if not config.get('dev', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = config['visible_devices']
        import torch
        try:
            assert torch.cuda.device_count() == config['n_gpu']
        except AssertionError as err:
            logging.error('Expected {} GPUs available, but only see {} (visible devices: {})'.format(config['n_gpu'],
                                                                                                    torch.cuda.device_count(),
                                                                                                    config[
                                                                                                        'visible_devices']))
            sys.exit(1)

        time = datetime.now().strftime("%m-%d-%H:%M")
        logging.info('Start time: {}'.format(time))

        logging.info('Number of GPUs available: {}'.format(torch.cuda.device_count()))
        logging.info('Using the following GPUs: {}'.format(config['visible_devices']))

        assert config['expected_batch_size'] == config['n_gpu'] * \
               config['training_arguments']['per_device_train_batch_size'] * \
               config['training_arguments']['gradient_accumulation_steps']

    

