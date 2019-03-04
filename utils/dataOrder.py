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

