import torch.nn as nn


class Embeddings(nn.Module):
	"""docstring for Embeddings"""
	def __init__(self, vocab_size, embedding_size, pre_trained_path='',word_to_index=None):
		super(Embeddings, self).__init__()
		self.embedding = nn.Embedding(vocab_size,embedding_size)
		if len(pre_trained_path) > 0:
			self.load_pretrained(pre_trained_path, word_to_index)
	
	def load_pretrained(self, pre_trained_path, word_to_index):
		gloves_vector = self.load_GloVe(pre_trained_path)

		assert type(word_to_index) == dict, 'The type of word_to_index is not supported.'
		
		for word in word_to_index
			if word.lower() in gloves_vector:
				indx = word_to_index[word]
				vec = torch.FloatTensor(gloves_vector[word])

				self.embedding.weight.data[idx,:].set_(vec)

	def load_GloVe(self, path):
		with open(path,'r') as f:
			vectors = f.readlines()
		word_vectors = {}
		for w in vectors:
			w = w.strip().split()
			word = w[0]
			vec = np.array(w[1:],dtype=float)
			word_vectors[word] = vec
		return word_vectors


	def forward(self, vec):
		out = self.embedding(vec)
		return out