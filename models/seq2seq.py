import torch.nn as nn

class Seq2Seq(nn.Module):
	"""docstring for Seq2Seq"""
	def __init__(self, encoder, decoder):
		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
	
	def forward(self,src,targer):
		encoded, memory = self.encoder(src)
		output = self.decoder(encoded, memory)
		return output

