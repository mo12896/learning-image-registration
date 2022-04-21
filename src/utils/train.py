



import wandb
wandb.init(project="registration", entity="mo12896")

import src.systemsetup as setup
from utils.params import Params









params = Params(setup.CONFIG_BASEMODEL)

wandb.config = {
  "learning_rate": params.learning_rate,
  "epochs": params.num_epochs,
  "batch_size": params.batch_size
}