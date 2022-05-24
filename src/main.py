import yaml
import sys
import torch
import os

from pprint import pprint
from argparse import ArgumentParser

import systemsetup as setup
from utils.modes import ExeModes
from utils.train import training_pipeline
from utils.test import inference_pipeline

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

mode_handler = {
    ExeModes.TRAIN.value: training_pipeline,
    ExeModes.TEST.value: inference_pipeline,
    # ExeModes.TRAIN_TEST.value: train_test_pipeline,
}


def main():
    """..."""
    """Define the config file to be used"""
    argparser = ArgumentParser(description="ImageRegistration")
    argparser.add_argument('--config',
                           type=str,
                           default="base_model",
                           help="The name of the model configuration. Supported:\n"
                                "- base_model")
    args = argparser.parse_args()
    config_file = args.config + '.yaml'
    config_path = os.path.join(setup.CONFIG_DIR, config_file)

    """Parse the config.yaml into a python dict"""
    with open(config_path, 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    """Validate the Execution Mode"""
    if configs['SETUP']['TRAIN'] and not configs['SETUP']['TEST']:
        mode = ExeModes.TRAIN.value
    if configs['SETUP']['TEST'] and not configs['SETUP']['TRAIN']:
        mode = ExeModes.TEST.value
    if configs['SETUP']['TRAIN'] and configs['SETUP']['TEST']:
        mode = ExeModes.TRAIN_TEST.value
    if not configs['SETUP']['TRAIN'] and not configs['SETUP']['TEST']:
        print("Please use either TRAIN or TEST or both.")
        return

    """Run the chosen pipeline"""
    pipeline = mode_handler[mode]
    pipeline(configs, exp_name=configs['META']['EXPERIMENT_NAME'],
             log_level=configs['SETUP']['LOGLEVEL'], notebook=False)


if __name__ == '__main__':
    main()
