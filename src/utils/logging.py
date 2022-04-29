import os
import logging
import functools
import sys

sys.path.append('../')
import systemsetup as setup

import wandb

wandb.init(project="registration", entity="mo12896")

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def log_model_tensorboard(avg_loss, avg_vloss):
	writer = SummaryWriter(setup.BASE_DIR + 'logs/tensorboard')
	# also: images, embeddings, model graph
	writer.add_scalars('Training vs. Validation Loss',
	                   {'Training': avg_loss, 'Validation': avg_vloss})
	writer.flush()
	writer.close()
