import os
import torch
import numpy as np
from torch.utils.data import Dataset

class Arxiv(Dataset):
	"""docstring for ArxivClassify"""
	def __init__(self, path, word_to_index=None, transform=None, pad_token='',limit=None):
		super(Arxiv, self).__init__()

		self.data = []
		self.transform = transform
		self.pad_token = pad_token
		self.path = path
		self.limit = limit
		self.__load_data__()
		print('from data: data loaded')
		self.word_to_index = word_to_index
		if self.word_to_index is None:
			pass
		elif isinstance(self.word_to_index,dict):
			self.__vectorize_data__()
		elif self.word_to_index == 'build':
			self.__build_dict__()
			self.__vectorize_data__()
		elif os.path.isfile(self.word_to_index):
			self.__load__dict()
			self.__vectorize_data__()
		else:
			print("ERROR")
	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		return self.data[idx]


	def __load_data__(self):
		with open(self.path,'r') as f:
			raw = f.read()
		
		docs = raw.split('\n\n')
		if self.limit is not None:
			docs = docs[:self.limit]
		self.data = []
		for doc in docs:
			sentences = doc.split('\n')
			lengths = []
			labels = []
			for i in range(len(sentences)):
				sentences[i] = sentences[i].strip().split()
				lengths.append(len(sentences[i]))
				labels.append(i)

				# if sentences[i][-1] == '.':
					# sentences[i] = sentences[i][:-1]
				
			self.data.append({'data':sentences, 'labels':labels, 'lengths':lengths})
		

		

	def __vectorize_word__(self,word):
		if word[-1] == ',':
			word = word[-1]
		if word in self.word_to_index:
			return self.word_to_index[word]
		else:
			return self.word_to_index['__unk__']

	def __vectorize_sentence__(self,sentence):
		for i in range(len(sentence)):
			sentence[i] = self.__vectorize_word__(sentence[i])
		return sentence

	def __vectorize_data__(self):
		for i in range(self.__len__()):
			for j in range(len(self.data[i]['data'])):
				self.data[i]['data'][j] = self.__vectorize_sentence__(self.data[i]['data'][j])
				self.data[i]['data'][j] = np.array(self.data[i]['data'][j])
	
	def __build_dict__(self):
		return
	
	def __load__dict(self):
		path = self.word_to_index
		with open(path,'r') as f:
			words = f.readlines()
		self.word_to_index = {}
		for i in range(len(words)):
			self.word_to_index[words[i].strip()] = i
