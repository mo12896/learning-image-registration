import torch


class Evaluator():
	def __init__(self, validation_set):
		self.validation_set = validation_set

	def evaluate(self, model):
		with torch.no_grad():
			for data in self.validation_set:
				fixed, moving, _ = data
				outputs = model(fixed, moving)

