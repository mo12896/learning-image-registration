import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


class Evaluator():
    def __init__(self, validation_set, eval_metric, device, batch_size):
        self.validation_set = validation_set
        self.eval_metric = eval_metric
        self.device = device
        self.batch_size = batch_size

    def validate(self, model):
        validation_loader = DataLoader(self.validation_set, batch_size=self.batch_size,
                                       shuffle=True)

        model.eval()
        running_loss = 0.
        final_loss = 0.
        batch_iter = tqdm(enumerate(validation_loader), 'Training',
                          total=len(validation_loader), leave=False)

        for i, (x, y, _) in batch_iter:
            image, label = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                pred = model(image)
                loss = self.eval_metric(pred, label)
                running_loss += loss.item()

                batch_iter.set_description(f'Validation: (loss {loss.item():.4f})')

                if i + 1 == len(validation_loader):
                    final_loss = running_loss / len(validation_loader)
                    print(f"The validation loss is: {final_loss}")
                    running_loss = 0.

        batch_iter.close()

        return final_loss
