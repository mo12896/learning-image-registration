import os
import logging
from tqdm import tqdm
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
from data.dataset import DatasetHandler
from utils.evaluate import Evaluator
from utils.logging import *


class Solver():
	def __init__(self, optimizer_class, optim_params, evaluator, criterion, device):
		self.optim_class = optimizer_class
		self.optim_params = optim_params
		self.optim = None  # defined for each training separately
		self.evaluator = evaluator
		self.criterion = criterion
		self.device = device

	def train(self,
	          model: torch.nn.Module,
	          training_set: torch.utils.data.Dataset,
	          n_epochs: int,
	          batch_size: int,
	          early_stop: bool,
	          eval_freq: int,
	          start_epoch: int,
	          save_models: bool):

		model.to(self.device)

		#Optimizer and lr scheduling
		self.optim.zero_grad()

		training_loader = DataLoader(training_set, batch_size=batch_size,
		                                shuffle=True, pin_memory=True)

		for epoch in tqdm(range(n_epochs)):
			print(f"EPOCH {epoch+1}: ")

			model.train(True)
			avg_loss = self.train_one_epoch(epoch, model, training_loader)

			if (epoch == start_epoch or
				epoch % eval_freq or
				epoch == n_epochs):

				model.train(False)
				val_results = self.evaluator.evaluate()
				#TODO: logic to log and safe best model

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
			self.optim.zero_grad()
			outputs = model(fixed, moving)
			loss = self.criterion(fixed, moving, outputs)
			loss.backward()
			optimizer.step()

			# log average loss
			running_loss += loss.item()
			if i % 1000 == 999:
				final_loss = running_loss / 1000
				log_train_losses(final_loss, epoch_index)
				print(f"The final loss of epoch {epoch_index} is: {final_loss}")
				running_loss = 0.

		return final_loss




	def compute_loss(self, model, data, iteration):


def training_pipeline(hyps: dict, log_level: str, exp_name):
	raw_data = setup.RAW_DATA_DIR + 'EMPIRE10/scans/'
	out_data = setup.INTERIM_DATA_DIR + 'EMPIRE10/scans/'
	ids = list(set([x.split('_')[0]
	                     for x in os.listdir(raw_data)]))
	partition = {}
	partition['train'], partition['validation'] = train_test_split(
		ids, test_size=0.33, random_state=42)

	#TODO: Create feasible transforms
	transform = transforms.Compose()

	# Generator
	training_set = DatasetHandler(partition['train'], root=out_data, transform=transform)

	init_logger(name, log_level, log_dir, mode)


	#TODO: implement!
	model = ModelHandler()
	evaluator = Evaluator()
	solver = Solver()
	solver.train(model, evaluator)

	finish_wandb_logger()