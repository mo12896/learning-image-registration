




import wandb
wandb.init(project="registration", entity="mo12896")

from utils.params import Params









params = Params("../configs/base_model.json")

wandb.config = {
  "learning_rate": params.learning_rate,
  "epochs": params.num_epochs,
  "batch_size": params.batch_size
}