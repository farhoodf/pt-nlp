import os
from io import open
import torch


class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)

class Corpus(object):
	def __init__(self, path):
		# print(os.path.join(path, 'train.raw'))
		self.dictionary = Dictionary()
		self.train = self.tokenize(os.path.join(path, 'train.raw'))
		self.valid = self.tokenize(os.path.join(path, 'valid.raw'))
		self.test = self.tokenize(os.path.join(path, 'test.raw'))

	def tokenize(self, path):
		"""Tokenizes a text file."""
		# print(path)
		assert os.path.exists(path)
		# Add words to the dictionary
		with open(path, 'r', encoding="utf8") as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)

		# Tokenize file content
		with open(path, 'r', encoding="utf8") as f:
			ids = torch.LongTensor(tokens)
			token = 0
			for line in f:
				words = line.split() + ['<eos>']
				for word in words:
					ids[token] = self.dictionary.word2idx[word]
					token += 1

		return ids