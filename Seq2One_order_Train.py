import argparse
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from models.SeqLabeling import RNNModel
from utils.dataOrder import ArxivBinary


import time


parser = argparse.ArgumentParser(description='Sequence Classification')

parser.add_argument('--data', metavar='DIR',help='path to dataset',default='../Order/Data/vec-test/')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('--log_interval', default=100, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--cpu', default=False, action='store_true',
					help='Using CPU instead of GPU.')
parser.add_argument('--clip', default=5, type=float,
					metavar='F', help='Gradient Clipping')


args = parser.parse_args()



def load_dataset():
	dataset = ArxivBinary(args.data)
	return dataset
def load_model():
	# RNN = RNNModel
	model = RNNModel(3,nlayers=2,nhid=1024,emdSize=1024)
	return model

def pad_data(s, maxlen):
	padded = np.zeros((maxlen,s.shape[1]), dtype=np.double)
	if len(s) > maxlen: padded[:] = s[:maxlen]
	else: padded[:len(s)] = s
	return padded

# def prepare_batch(x,y):
#     x.sort(key=lambda item: -item.shape[1])
#     x_len = []
#     for i in range(len(x)):
#         l = min(x[i].shape[1],50)
#         x[i] = pad_data(x[i][0],50)
#         x_len.append(l)
#     x = np.array(x)
#     x_len = np.array(x_len)
#     x = torch.from_numpy(x).float().cuda()
#     x_len = torch.from_numpy(x_len).float()
#     y = torch.from_numpy(np.array(y,dtype=int)).cuda()
#     return x, x_len, y

def prepare_batch(x):
	x = x[:,2,:,:]
	return x.transpose(0,1)
def train_epoch(model, dataset, device, lr,epoch = 1):
	dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=4)
	loss_fn = nn.CrossEntropyLoss()
	total_epoch_loss = 0
	total_epoch_acc = 0
	# model.cuda()
#     optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = 0.001)
	optim = torch.optim.Adam(model.parameters(), lr=lr)
#     optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	steps = 0
	model.train()
	for i_batch, batch in enumerate(dataloader):
		x = prepare_batch(batch['data'].to(device=device))
		y = batch['label'].to(device=device)
		lengths = batch['lengths']
		# x,x_len,y = prepare_batch(x,y)
		# y = torch.autograd.Variable(y).long()
#         optim.zero_grad()
		model.zero_grad()
		y_hat = model(x,lengths)
		loss = loss_fn(y_hat,y)
		num_currect = (torch.max(y_hat, 1)[1].view(y.size()).data == y.data).float().sum()
		acc = 100.0 * num_currect/args.batch_size
		loss.backward()
		optim.step()
		
		steps += 1
		total_epoch_loss += loss.item()
		total_epoch_acc += acc.item()
		
		if steps % args.log_interval == 0:
			print (f'Epoch: {epoch}, batch: {steps}, Training Loss: {total_epoch_loss/steps:.4f}, Training Accuracy: {total_epoch_acc/steps: .2f}%')
			
		
	return total_epoch_loss/steps, total_epoch_acc/steps

def eval(model,dataset,device):
	dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=False, num_workers=4)
	loss_fn = nn.CrossEntropyLoss()
	total_epoch_loss = 0
	total_epoch_acc = 0
	# model.cuda()
#     optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = 0.001)
#     optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	steps = 0
	model.eval() # check this
	with torch.no_grad():
		for i_batch, batch in enumerate(dataloader):
			x = prepare_batch(batch['data'].to(device=device))
			y = batch['label'].to(device=device)
			lengths = batch['lengths']
			# x,x_len,y = prepare_batch(x,y)
			# y = torch.autograd.Variable(y).long()
	#         optim.zero_grad()
			# model.zero_grad()
			y_hat = model(x,lengths)
			loss = loss_fn(y_hat,y)
			num_currect = (torch.max(y_hat, 1)[1].view(y.size()).data == y.data).float().sum()
			acc = 100.0 * num_currect/args.batch_size
			# loss.backward()
			# optim.step()
			
			steps += 1
			total_epoch_loss += loss.item()
			total_epoch_acc += acc.item()
		
		# if steps % args.log_interval == 0:
			# print (f'Epoch: {epoch}, batch: {steps}, Training Loss: {total_epoch_loss/steps:.4f}, Training Accuracy: {total_epoch_acc/steps: .2f}%')
		return total_epoch_loss/steps, total_epoch_acc/steps

def train(model,dataset,device):

	try:
		for epoch in range(1, args.epochs+1):
			epoch_start_time = time.time()
			loss,acc = train_epoch(model,dataset,epoch=epoch,device=device,lr=args.lr)

			# val_loss = evaluate(val_data)
			print('-' * 89)
			print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
					'train accuracy {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
											   loss, acc))
			print('-' * 89)
			# Save the model if the validation loss is the best we've seen so far.
			# if not best_val_loss or val_loss < best_val_loss:
			# 	# with open(save, 'wb') as f:
			# 	#     torch.save(model, f)
			# 	best_val_loss = val_loss
			# else:
			# 	# Anneal the learning rate if no improvement has been seen in the validation dataset.
			# 	lr /= 4.0
	except KeyboardInterrupt:
		print('-' * 89)
		print('Exiting from training early')


def main():
	device = 'cpu'
	if not args.cpu and torch.cuda.is_available():
		device = 'cuda'


	dataset = load_dataset()
	model = load_model().to(device=device)
	# print(model)
	train(model,dataset,device)

if __name__ == '__main__':
	main()