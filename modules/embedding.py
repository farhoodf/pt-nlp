import torch.nn as nn


class Embeddings(nn.Module):
	"""docstring for Embeddings"""
	def __init__(self, vocab_size, embedding_size):
		super(Embeddings, self).__init__()
		self.embedding = nn.Embedding(vocab_size,embedding_size)
		
	def load_pretrained(self):
		# To write
		return None

	def forward(self, vec):
		out = self.embedding(vec)
		return out