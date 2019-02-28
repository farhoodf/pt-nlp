import torch.nn as nn
import torch.nn.functional as F
from modules import RNNModule


class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, ntoken, noutput, rnn_type='GRU', emdSize=256, nhid=128, nlayers=1, dropout=0.5, tie_weights=False, embedding=None):
		super(RNNModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		if embedding is None:
			self.encoder = nn.Embedding(ntoken, emdSize)
		else:
			self.encoder = embedding

		
		# if rnn_type in ['LSTM', 'GRU']:
		# 	self.rnn = getattr(nn, rnn_type)(emdSize, nhid, nlayers, dropout=dropout)
		# else:
		# 	# try:
		# 	#     nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
		# 	# except KeyError:
		# 	raise ValueError( """An invalid option for `--model` was supplied,
		# 						 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
		# 	# self.rnn = nn.RNN(emdSize, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		self.rnn = RNNModule.RNNModule(rnn_type='GRU', ninp=emdSize, nhid=nhid, nlayers=nlayers, dropout=dropout)
		self.decoder = nn.Linear(nhid, noutput)

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462

		self.init_weights(embedding)

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

		# self.softmax = 

	def init_weights(self, embedding):
		initrange = 0.1
		if embedding is None:
			self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, x, lengths):
		_, batch_size = x.size()
		#initial hidden states
		hidden = self.init_hidden(batch_size)
		x = self.drop(self.encoder(x))


		x = nn.utils.rnn.pack_padded_sequence(x, lengths)

		x, hidden = self.rnn(x, hidden)

		x, l = nn.utils.rnn.pad_packed_sequence(x)
		# x = self.drop(x)


		decoded = self.decoder(hidden[0])
		# decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
		decoded = F.softmax(decoded,dim=-1)
		return decoded

	def init_hidden(self, bsz):
		return self.rnn.init_hidden(bsz)
