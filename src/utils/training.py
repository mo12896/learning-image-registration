import os
import logging
import json

import torch
from torch.utils.data import DataLoader
import torch.optim

import sys
sys.path.append('../')
import systemsetup as setup
from utils.params import Params
from utils.evaluate import Evaluator

params = Params(setup.CONFIG_BASEMODEL)


class Solver():
	def __init__(self, optimizer_class, optim_params, evaluator, loss_fn, device):
		self.optim_class = optimizer_class
		self.optim_params = optim_params
		self.optim = None  # defined for each training separately
		self.evaluator = evaluator

	def train(self,
	          model: torch.nn.Module,
	          training_set: torch.utils.data.Dataset,
	          n_epochs: int,
	          batch_size: int,
	          early_stop: bool,
	          eval_freq: int,
	          start_epoch: int,
	          save_models:):

		model.to(self.device)

		#Optimizer and lr scheduling
		self.optim.zero_grad()

		training_loader = DataLoader(training_set, batch_size=batch_size,
		                                shuffle=True, pin_memory=True)

		for epoch in range(n_epochs):
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

		# Save final model
		if save_models:
			raise NotImplementedError


	def train_one_epoch(self, epoch_index, model, training_loader):
		running_loss = 0.
		final_loss = 0.

		for i, data in enumerate(training_loader):
			fixed, moving, _ = data
			self.optim.zero_grad()
			outputs = model(fixed, moving)
			loss = loss_fn(fixed, moving, outputs)
			loss.backward()
			optimizer.step()

			# log average loss
			running_loss += loss.item()
			if i % 1000 == 999:
				final_loss = running_loss / 1000
				print(f"The final loss of epoch {epoch_index} is: {final_loss}")
				running_loss = 0.

		return final_loss




	def compute_loss(self, model, data, iteration):

