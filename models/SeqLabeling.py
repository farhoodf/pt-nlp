import torch.nn as nn
import torch.nn.functional as F
from modules import RNNModule, Dummy, Embedding


class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, noutput, rnn_type='GRU',ntoken = 0 ,emdSize=256, nhid=256, nlayers=1, dropout=0.5, tie_weights=False, embedding=None):
		super(RNNModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		init_embed = False
		if embedding is None:
			if ntoken == 0:
				self.encoder = Dummy.Dummy()
			else:
				self.encoder = Embedding.Embeddings(ntoken, emdSize)
				init_embed = True
		elif issubclass(type(embedding),nn.Module):
			self.encoder = embedding
		else:
			raise ValueError('The type of Embedding is not defined')

		
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

		self.init_weights(init_embed)

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

		# self.softmax = 

	def init_weights(self, init_embed):
		initrange = 0.1
		if init_embed:
			self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, x, lengths):
		batch_size = x.size()[1]
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
