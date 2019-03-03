import torch.nn as nn

class Dummy(nn.Module):
	"""docstring for Dummy"""
	def __init__(self):
		super(Dummy, self).__init__()
	def forward(self, x):
		return x