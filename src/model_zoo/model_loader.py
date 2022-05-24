


class ModelLoader():
	def __init__(self, class_name: str):
		self.class_name = class_name
		self.instantiate_from_class_name()

	def instantiate_from_class_name(self):
		dynamic_class = globals[self.class_name]
		return dynamic_class()