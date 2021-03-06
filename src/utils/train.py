import os
import logging
from copy import deepcopy
import yaml

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from transforms.tensor_transforms import Create2D, Rescale, AddChannel, NormalizeSample
from transforms.transforms_handler import *
from transforms.image_transforms import RandomVerticalFlip
from utils.eval_metrics import DiceLoss
from model_zoo.model_loader import ModelLoader
from model_zoo.unet import UNet
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
                 save_models: bool = True):

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
        self.best_val_loss = 10000000

    def train(self):
        self.model.to(self.device)

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for epoch in progressbar:
            print(f"EPOCH {epoch + 1}: ")
            logger = {}

            """ Training Block """
            avg_loss = self._train_one_epoch(epoch)
            logger['TrainingLoss'] = avg_loss

            """ Validation Block """
            if (epoch % self.eval_freq or
                    epoch == self.epochs):
                current_val_loss = self.evaluator.validate(self.model, epoch)
                self.lr_scheduler.step(current_val_loss)

                logger['ValidationLoss'] = current_val_loss
                logger['Learning Rate'] = self.optimizer.param_groups[0]['lr']

                # Save best model
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    print(f"\nCurrent best validation loss: {self.best_val_loss}")
                    print(f"\nSaving best model for epoch: {epoch + 1}\n")
                    if self.save_models:
                        torch.save(self.model.state_dict(),
                                   os.path.join(setup.BASE_DIR, 'models/best.model'))

                log_stuff(logger, epoch)

            # Perform early stopping based on criteria:
            if self.early_stop:
                raise NotImplementedError

        # Save final model (use wandb.save() in .onnx format)
        if self.save_models:
            log_best_model()

    def _train_one_epoch(self, epoch_index):
        self.model.train()
        running_loss = 0.
        final_loss = {}

        batch_iter = enumerate(self.training_loader)
        for i, (x, y, _) in batch_iter:
            image, label = (x.to(self.device), y.to(self.device))
            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.criterion(pred, label)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            if i + 1 == len(self.training_loader):
                final_loss = running_loss / len(self.training_loader)
                print(f"\nThe final loss of epoch {epoch_index} is: {final_loss}")
                running_loss = 0.

        return final_loss


def training_pipeline(configs: dict, log_level: str, notebook: bool, exp_name: str):
    """..."""
    """Setup hyperparameters"""
    train_setup = {k.lower(): v for k, v in configs['SETUP'].items()}
    hyper = {k.lower(): v for k, v in configs['HYPERPARAMETERS'].items()}

    raw_data = setup.RAW_DATA_DIR + 'EMPIRE10/scans/'
    dataset = train_setup['dataset']
    random_seed = train_setup['random_seed']
    test_size = train_setup['split']
    patience = hyper['patience']

    """Initalize Logger """
    folders = [setup.MODEL_DIR, setup.LOG_DIR, setup.WANDB_DIR]
    clean_logger(folders)

    init_logger(ExeModes.TRAIN.name, log_level, setup.LOG_DIR, mode=ExeModes.TRAIN)
    train_logger = logging.getLogger(ExeModes.TRAIN.name)
    train_logger.info("Start training '%s'...")

    """Setup Data Generator"""
    # TODO: different splitter for segmentation and registration!?
    ids = list(set([x.split('_')[0] for x in os.listdir(raw_data)]))
    partition = {'train': (train_test_split(
        ids, test_size=test_size, random_state=random_seed))[0], 'validation': (train_test_split(
        ids, test_size=test_size, random_state=random_seed))[1]}

    pre_transform = transforms.Compose([
        Create2D('y'),
        AddChannel(axs=0),
        Rescale((256, 256))
    ])

    # TODO: use separate transforms for validation!
    transform = ComposeDouble([
        FunctionWrapperDouble(RandomVerticalFlip(prob=0.5)),
    ])

    """Data Generator"""
    training_set = SegmentationDataset(partition['train'], dataset=dataset,
                                       transform=transform, use_cache=True,
                                       pre_transform=pre_transform)
    validation_set = SegmentationDataset(partition['validation'], dataset=dataset,
                                         transform=transform, use_cache=True,
                                         pre_transform=pre_transform)

    training_loader = DataLoader(training_set, batch_size=hyper['batch_size'], shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=hyper['batch_size'],
                                   shuffle=True)

    """Training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet()
    # train_logger.info(f"{model.count_parameters()} parameters in the model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=hyper['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience)
    eval_metric = DiceLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    evaluator = Evaluator(validation_loader=validation_loader,
                          eval_metric=eval_metric,
                          device=device,
                          notebook=notebook)

    solver = Solver(model=model,
                    device=device,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    evaluator=evaluator,
                    criterion=eval_metric,
                    training_loader=training_loader,
                    notebook=notebook,
                    epochs=hyper['epochs'],
                    eval_freq=hyper['eval_every'], )

    solver.train()

    finish_wandb_logger()
