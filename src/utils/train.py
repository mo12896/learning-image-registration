import os
import logging
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

import sys

sys.path.append('../')
import systemsetup as setup
from data.dataset import DatasetHandler
from utils.evaluate import Evaluator
from utils.modes import ExeModes
from features.tensor_transforms import Create2D, Rescale, AddChannel, NormalizeSample
from utils.eval_metrics import DiceLoss
from models.model_loader import ModelLoader
from models.unet import UNet
from utils.logging import *


class Solver():
    def __init__(self, optimizer, evaluator, criterion, device):
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.criterion = criterion
        self.device = device

    def train(self,
              model: torch.nn.Module,
              training_set: torch.utils.data.Dataset,
              n_epochs: int,
              batch_size: int,
              early_stop: False,
              eval_freq: int,
              start_epoch: int,
              save_models: False):

        model.to(self.device)

        # Optimizer and lr scheduling
        self.optimizer.zero_grad()

        training_loader = DataLoader(training_set, batch_size=batch_size,
                                     shuffle=True, pin_memory=True)

        for epoch in tqdm(range(n_epochs)):
            print(f"EPOCH {epoch + 1}: ")

            model.train(True)
            avg_loss = self.train_one_epoch(epoch, model, training_loader)

            if (epoch == start_epoch or
                    epoch % eval_freq or
                    epoch == n_epochs):
                model.train(False)
                val_results = self.evaluator.evaluate()
            # TODO: logic to log and safe best model

            # Save intermediate model after each epoch
            if save_models:
                raise NotImplementedError

            # Perform early stopping based on criteria:
            if early_stop:
                raise NotImplementedError

        # Save final model (use wandb.save() in .onnx format)
        if save_models:
            raise NotImplementedError

    def train_one_epoch(self, epoch_index, model, training_loader):
        running_loss = 0.
        final_loss = 0.

        for i, data in enumerate(training_loader):
            fixed, moving, _ = data
            fixed, moving = fixed.to(self.device), moving.to(self.device)
            self.optimizer.zero_grad()
            outputs = model(fixed, moving)
            loss = self.criterion(fixed, moving, outputs)
            loss.backward()
            self.optimizer.step()

            # log average loss
            running_loss += loss.item()
            if i % 1000 == 999:
                final_loss = running_loss / 1000
                log_train_losses(final_loss, epoch_index)
                print(f"The final loss of epoch {epoch_index} is: {final_loss}")
                running_loss = 0.

        return final_loss

    def compute_loss(self, model, data, iteration):
        pass


def training_pipeline(hyper: dict, log_level: str, exp_name: str):
    train_setup = {k.lowercase: v for k, v in hyper['SETUP']}
    hyper = {k.lowercase: v for k, v in hyper['HYPERPARAMETERS']}

    raw_data = setup.RAW_DATA_DIR + 'EMPIRE10/scans/'
    dataset = train_setup['dataset']
    task = train_setup['seg']
    ids = list(set([x.split('_')[0] for x in os.listdir(raw_data)]))

    partition = {'train': (train_test_split(
        ids, test_size=0.33, random_state=42))[0], 'validation': (train_test_split(
        ids, test_size=0.33, random_state=42))[1]}

    shape = (97, 97)
    transform = transforms.Compose([
        # Data Preprocessing
        Create2D('y'),
        AddChannel(axs=0),
        Rescale(shape)
    ])

    init_logger(ExeModes.TRAIN.name, log_level, setup.LOG_DIR, mode=ExeModes.TRAIN)
    train_logger = logging.getLogger(ExeModes.TRAIN.name)
    train_logger.info("Start training '%s'...")

    # Data Generator
    training_set = DatasetHandler(partition['train'], dataset=dataset, task=task,
                                  transform=transform)
    validation_set = DatasetHandler(partition['validation'], dataset=dataset,
                                    task=task, transform=transform)

    # Training
    # TODO: implement!
    model = UNet()
    train_logger.info("%d parameters in the model.", model.count_parameters())
    print("hi")

    optimizer = optim.Adam(model.parameters(), lr=hyper['learning_rate'])
    evaluator = Evaluator(validation_set=validation_set,
                          eval_metrics=DiceLoss)

    solver = Solver(optimizer=optimizer, evaluator=evaluator, criterion=criterion,
                    device=train_setup['device'])

    solver.train(model=model,
                 training_set=training_set,
                 n_epochs=hyper['epochs'],
                 batch_size=hyper['batch_size'],
                 eval_freq=hyper['eval_every'],
                 start_epoch=hyper['start_epoch'])

    finish_wandb_logger()
