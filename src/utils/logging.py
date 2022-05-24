import os
import logging
import functools
import sys

sys.path.append('../')

import wandb
import torch
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import systemsetup as setup
from utils.modes import ExeModes
from models.unet import UNet

use_wandb = True
wandb_run = None


def init_wandb_logger():
    global wandb_run
    global use_wandb
    if use_wandb:
        wandb_run = wandb.init()


def finish_wandb_logger():
    global wandb_run
    if use_wandb:
        wandb_run.finish()


def init_basic_logger(name, log_level, log_dir, mode):
    logger = logging.getLogger(name)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Config
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logger.setLevel(numeric_level)

    # File Logger
    file_formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
    log_file = os.path.join(log_dir, mode.name.lower() + ".log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Logger
    console_formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


def init_logger(logger_name: str, log_dir: str, log_level: str, mode: ExeModes):
    init_basic_logger(logger_name, log_dir, log_level, mode)

    if mode == ExeModes.TRAIN:
        init_wandb_logger()


def log_stuff(loss, iteration):
    train_logger = logging.getLogger()
    train_logger.info(loss)

    if use_wandb:
        wandb.log(loss, step=iteration)


def log_best_model():
    test_input = torch.randn(size=(1, 1, 512, 512), dtype=torch.float32)
    final_model = UNet()
    final_model.load_state_dict(
        torch.load(os.path.join(setup.BASE_DIR, 'models/best.model')))
    torch.onnx.export(final_model, test_input, os.path.join(setup.BASE_DIR, 'models/best.onnx'))
    wandb.save("best.onnx")

# def log_model_tensorboard(avg_loss, avg_vloss):
# 	writer = SummaryWriter(setup.BASE_DIR + 'logs/tensorboard')
# 	# also: images, embeddings, model graph
# 	writer.add_scalars('Training vs. Validation Loss',
# 	                   {'Training': avg_loss, 'Validation': avg_vloss})
# 	writer.flush()
# 	writer.close()
