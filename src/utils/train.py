import os
import logging
import yaml

import torch
from torch.utils.data import DataLoader
import torch.optim

import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

import sys

sys.path.append('../')
import systemsetup as setup
from data.dataset import SegmentationDataset, RegistrationDataset
from utils.evaluate import Evaluator
from utils.modes import ExeModes
from features.tensor_transforms import Create2D, Rescale, AddChannel, NormalizeSample
from utils.eval_metrics import DiceLoss
from models.model_loader import ModelLoader
from models.unet import UNet
from utils.logging import *


class Solver():
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 evaluator: Evaluator,
                 criterion: torch.nn.Module,
                 training_loader: torch.utils.data.DataLoader,
                 epochs: int,
                 eval_freq: int,
                 notebook: bool,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 early_stop: bool = False,
                 save_models: bool = False):

        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.criterion = criterion
        self.training_loader = training_loader
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.notebook = notebook
        self.lr_scheduler = lr_scheduler
        self.early_stop = early_stop
        self.save_models = save_models

    def train(self):
        self.model.to(self.device)

        # Optimizer and lr scheduling ...

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for epoch in progressbar:
            print(f"EPOCH {epoch + 1}: ")

            """ Training Block """
            avg_loss = self._train_one_epoch(epoch, self.training_loader)

            """ Validation Block """
            if (epoch % self.eval_freq or
                    epoch == self.epochs):
                val_results = self.evaluator.validate(self.model)

            # Save intermediate model after each epoch
            if self.save_models:
                raise NotImplementedError

            # Perform early stopping based on criteria:
            if self.early_stop:
                raise NotImplementedError

        # Save final model (use wandb.save() in .onnx format)
        if self.save_models:
            raise NotImplementedError

    def _train_one_epoch(self, epoch_index, training_loader):
        self.model.train()
        running_loss = 0.
        final_loss = {}

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        # batch_iter = tqdm(enumerate(training_loader), 'Training',
        #                  total=len(training_loader), leave=False)
        batch_iter = enumerate(training_loader)

        for i, (x, y, _) in batch_iter:
            image, label = (x.to(self.device), y.to(self.device))
            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.criterion(pred, label)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # batch_iter.set_description(f'Training: (loss {loss.item():.4f})')

            if i + 1 == len(training_loader):
                final_loss['total_loss'] = running_loss / len(training_loader)
                log_train_losses(final_loss, epoch_index)
                print(f"The final loss of epoch {epoch_index} is: {final_loss}")
                running_loss = 0.

        # batch_iter.close()

        return final_loss


def training_pipeline(hyper: dict, log_level: str, notebook: bool, exp_name: str):
    """..."""
    """Setup hyperparameters"""
    train_setup = {k.lower(): v for k, v in hyper['SETUP'].items()}
    hyper = {k.lower(): v for k, v in hyper['HYPERPARAMETERS'].items()}

    raw_data = setup.RAW_DATA_DIR + 'EMPIRE10/scans/'
    dataset = train_setup['dataset']
    task = train_setup['task']
    ids = list(set([x.split('_')[0] for x in os.listdir(raw_data)]))
    random_seed = 42
    test_size = 0.33

    """Initalize Logger """
    init_logger(ExeModes.TRAIN.name, log_level, setup.LOG_DIR, mode=ExeModes.TRAIN)
    train_logger = logging.getLogger(ExeModes.TRAIN.name)
    train_logger.info("Start training '%s'...")

    """Setup Data Generator"""
    partition = {'train': (train_test_split(
        ids, test_size=test_size, random_state=random_seed))[0], 'validation': (train_test_split(
        ids, test_size=test_size, random_state=random_seed))[1]}

    shape = (256, 256)
    pre_transform = transforms.Compose([
        Create2D('y'),
        AddChannel(axs=0),
        Rescale(shape)
    ])

    # transform = transforms.Compose([
    #     Rescale(shape)
    # ])

    """Data Generator"""
    training_set = SegmentationDataset(partition['train'], dataset=dataset,
                                       transform=None, use_cache=True,
                                       pre_transform=pre_transform)
    validation_set = SegmentationDataset(partition['validation'], dataset=dataset,
                                         transform=None, use_cache=True,
                                         pre_transform=pre_transform)

    training_loader = DataLoader(training_set, batch_size=hyper['batch_size'], shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=hyper['batch_size'],
                                   shuffle=True)

    """Training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet()
    # train_logger.info(f"{model.count_parameters()} parameters in the model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=hyper['learning_rate'])
    eval_metric = DiceLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    evaluator = Evaluator(validation_loader=validation_loader,
                          eval_metric=eval_metric,
                          device=device,
                          notebook=notebook)

    solver = Solver(model=model,
                    device=device,
                    optimizer=optimizer,
                    evaluator=evaluator,
                    criterion=eval_metric,
                    training_loader=training_loader,
                    notebook=notebook,
                    epochs=hyper['epochs'],
                    eval_freq=hyper['eval_every'])

    solver.train()

    finish_wandb_logger()
