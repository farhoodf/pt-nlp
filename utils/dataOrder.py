import os
import torch
import numpy as np
from torch.utils.data import Dataset



class ArxivBinary(Dataset):
	"""docstring for ArxivBinary"""
	def __init__(self, root_dir, transform=None, preload=False ,maxlen=30):
		super(ArxivBinary, self).__init__()

		self.fnames = os.listdir(root_dir)
		self.root_dir = root_dir
		self.transform = transform
		self.preload = preload
		self.maxlen = maxlen

		if self.preload:
			self.data = []
			for i in range(self.__len__()):
				self.data.append(self.__readitem__(i))
	
	def __len__(self):
		return len(self.fnames)


	def __getitem__(self, idx):
		if self.preload:
			return self.data[idx]
		else:
			return self.__readitem__(idx)


	def __readitem__(self,idx):
		data = self.__pad__(np.load(self.root_dir+self.fnames[idx]))
		label = int(self.fnames[idx].split('-')[0])

		if self.transform:
			data = self.transform(data)

		return {'data':data,'label':label, 'lengths':len(data)}

	def __pad__(self,data):
		shape = list(data.shape)
		shape[1] = self.maxlen
		padded = np.zeros((shape), dtype=data.dtype)
		if data.shape[1] > self.maxlen: 
			padded[:] = data[:,:self.maxlen,:]
		else: 
			padded[:,:data.shape[1],:] = data
		return padded

class ArxivClassify(Dataset):
	"""docstring for ArxivClassify"""
	def __init__(self, path, word_to_index=None, transform=None, maxlen=30, pad_token=''):
		super(ArxivClassify, self).__init__()

		self.data = []
		self.transform = transform
		self.maxlen = maxlen
		self.pad_token = pad_token
		self.path = path
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
			data = f.read()
		
		docs = data.split('\n\n')
		
		data = []
		for doc in docs:
			sentences = doc.split('\n')
			for i in range(len(sentences)):
				sent = sentences[i].strip()
				if i == 0:
					continue
				elif i == 1:
					l = 0
				elif i == len(sentences)-1:
					l = 2
				else:
					if np.random.rand() > 0.3:
						continue
					l = 1
				if sent[-1] == '.':
					sent = sent[:-1]

				tokenized = sent.split()
				tokenized = self.__add_pad__(tokenized)
				
				data.append({'data':tokenized, 'label':l, 'lengths':len(tokenized)})
		
		self.data = data
		

	# def __readitem__(self,idx):
	# 	data = self.__add_pad__(np.load(self.root_dir+self.fnames[idx]))
	# 	label = int(self.fnames[idx].split('-')[0])

	# 	if self.transform:
	# 		data = self.transform(data)

	# 	return {'data':data,'label':label, 'lengths':len(data)}

	def __add_pad__(self,data):

		padded = [self.pad_token]*self.maxlen
		if len(data) > self.maxlen: 
			padded[:] = data[:self.maxlen]
		else: 
			padded[:len(data)] = data
		return padded

	def __to_vector__(self,word):
		if word[-1] == ',':
			word = word[-1]
		if word in self.word_to_index:
			return self.word_to_index[word]
		else:
			return self.word_to_index['__unk__']


		return vectorized
	def __vectorize_data__(self):
		for i in range(self.__len__()):
			for j in range(len(self.data[i]['data'])):
				self.data[i]['data'][j] = self.__to_vector__(self.data[i]['data'][j])
			self.data[i]['data'] = np.array(self.data[i]['data'])
			# print(self.data[i]['data'])
	def __build_dict__(self):
		return
	def __load__dict(self):
		path = self.word_to_index
		with open(path,'r') as f:
			words = f.readlines()
		self.word_to_index = {}
		for i in range(len(words)):
			self.word_to_index[words[i].strip()] = i
		# print(self.word_to_index)

