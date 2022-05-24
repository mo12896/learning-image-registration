import torch
from torch.utils.data import DataLoader

from utils.logging import *


class Evaluator():
    def __init__(self, validation_loader, eval_metric, device, notebook):
        self.validation_loader = validation_loader
        self.eval_metric = eval_metric
        self.device = device
        self.notebook = notebook

    def validate(self, model, lr_scheduler, epoch_index):
        model.eval()
        running_loss = 0.
        val_loss = 0.

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        batch_iter = tqdm(enumerate(self.validation_loader), 'Training',
                          total=len(self.validation_loader), leave=False)

        for i, (x, y, _) in batch_iter:
            image, label = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                pred = model(image)
                loss = self.eval_metric(pred, label)
                running_loss += loss.item()

                batch_iter.set_description(f'Validation: (loss {loss.item():.4f})')

                if i + 1 == len(self.validation_loader):
                    val_loss = running_loss / len(self.validation_loader)
                    lr_scheduler.step(val_loss)
                    print(f"\nThe validation loss of epoch {epoch_index} is: {val_loss}")
                    running_loss = 0.

        batch_iter.close()

        return val_loss
