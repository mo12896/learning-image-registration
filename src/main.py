import yaml
import sys
import torch
import os

from pprint import pprint

import systemsetup as setup
from utils.modes import ExeModes
from utils.train import training_pipeline

configs = setup.CONFIG_DIR + 'base_model.yaml'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

with open(configs, 'r') as stream:
    try:
        hyper = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

mode_handler = {
    ExeModes.TRAIN.value: training_pipeline,
    # ExeModes.TEST.value: test_pipeline,
    # ExeModes.TRAIN_TEST.value: train_test_pipeline,
}


def main(hyper):
    if hyper['SETUP']['TRAIN'] and not hyper['SETUP']['TEST']:
        mode = ExeModes.TRAIN.value
    if hyper['SETUP']['TEST'] and not hyper['SETUP']['TRAIN']:
        mode = ExeModes.TEST.value
    if hyper['SETUP']['TRAIN'] and hyper['SETUP']['TEST']:
        mode = ExeModes.TRAIN_TEST.value
    if not hyper['SETUP']['TRAIN'] and not hyper['SETUP']['TEST']:
        print("Please use either TRAIN or TEST or both.")
        return

    # Run
    pipeline = mode_handler[mode]
    pipeline(hyper, exp_name=hyper['META']['EXPERIMENT_NAME'],
             log_level=hyper['SETUP']['LOGLEVEL'], notebook=False)


if __name__ == '__main__':
    main(hyper)
