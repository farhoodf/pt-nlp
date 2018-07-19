import torch.nn as nn

class Attention(nn.Module):
	"""docstring for Attention"""
	def __init__(self, arg):
		super(Attention, self).__init__()
		self.arg = arg
		self.softmax = nn.Softmax(-1)
		self.prjEnergy = nn.Linear(,)
		self.prjState = nn.Linear(,)
		self.prjMemory = nn.Linear(,)
		self.tanh = nn.Tanh()
	def forward(self, state, memory):
		pState = self.prjState(state)
		pMemory = self.prjMemory(memory)
		energy = self.prjEnergy(self.tanh(torch.add(pState,pMemory)))
		attn = self.softmax(energy)
		return attn
		