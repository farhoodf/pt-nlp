import argparse
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from code import data, model


parser = argparse.ArgumentParser(description='Sequence Classification')

parser.add_argument('data', metavar='DIR',help='path to dataset',default='./Data/4-15-divided-tokenized-vectored/test/')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log_interval', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--cpu', default=False, action='store_true',
                    help='Using CPU instead of GPU.')
parser.add_argument('--clip', default=5, type=float,
                    metavar='F', help='Gradient Clipping')


args = parser.parse_args()



def load_dataset():

def load_model():
	RNN = model.RNN
	rnn = RNN(1024,3,nb_layers=2,nb_units=1024)
	rnn.cuda()

def pad_data(s, maxlen):
    padded = np.zeros((maxlen,s.shape[1]), dtype=np.double)
    if len(s) > maxlen: padded[:] = s[:maxlen]
    else: padded[:len(s)] = s
    return padded

def prepare_batch(x,y):
    x.sort(key=lambda item: -item.shape[1])
    x_len = []
    for i in range(len(x)):
        l = min(x[i].shape[1],50)
        x[i] = pad_data(x[i][0],50)
        x_len.append(l)
    x = np.array(x)
    x_len = np.array(x_len)
    x = torch.from_numpy(x).float().cuda()
    x_len = torch.from_numpy(x_len).float()
    y = torch.from_numpy(np.array(y,dtype=int)).cuda()
    return x, x_len, y


def train_epoch(model, batch_gen, batch_size, ds, epoch = 1):
    loss_fn = nn.CrossEntropyLoss()
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
#     optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = 0.001)
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
#     optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    steps = 0
    model.train()
    for x, y in batch_gen(ds, batch_size=batch_size):
        x,x_len,y = prepare_batch(x,y)
        y = torch.autograd.Variable(y).long()
#         optim.zero_grad()
        model.zero_grad()
        y_hat = model(x,x_len)
        loss = loss_fn(y_hat,y)
        num_currect = (torch.max(y_hat, 1)[1].view(y.size()).data == y.data).float().sum()
        acc = 100.0 * num_currect/batch_size
        loss.backward()
        optim.step()
        
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, batch: {steps}, Training Loss: {total_epoch_loss/steps:.4f}, Training Accuracy: {total_epoch_acc/steps: .2f}%')
            
        
    return total_epoch_loss/steps, total_epoch_acc/steps



def train():
	for i in range(5):
    

	try:
		for epoch in range(1, epochs+1):
			epoch_start_time = time.time()
			loss,acc = train_epoch(rnn,get_batch,64,ds,epoch)

			# val_loss = evaluate(val_data)
			print('-' * 89)
			print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
					'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
											   val_loss, math.exp(val_loss)))
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