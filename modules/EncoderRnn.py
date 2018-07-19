
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class EncoderRNN(nn.Module):
	"""docstring for EncoderRNN"""
	def __init__(self, num_layers, hidden_size, embeddings=None,
				rnn_type='GRU', bidirectional=False,
				dropout=0.0, use_bridge=False):
		super(EncoderRNN, self).__init__()

		# num_directions = 2 if bidirectional else 1
		# hidden_size = hidden_size // num_directions

		assert embeddings is not None
		self.embeddings = embeddings

		self.rnn = getattr(nn, rnn_type)(input_size=embeddings.embedding_size,
										hidden_size=hidden_size,
										num_layers=num_layers,
										dropout=dropout,
										bidirectional=bidirectional)
		self.no_pack_padded_seq = False
		# self.use_bridge = use_bridge
		# if self.use_bridge:
		# 	self._initialize_bridge(rnn_type,
		# 							hidden_size,
		# 							num_layers)

	def forward(self, input, lengths=None):
		embedded = self.embeddings(input)
		packed_emb = embedded

		
		if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(embedded, lengths)

        # Are you sure???? out,hid or hid,out?
        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        # if self.use_bridge:
        #     encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank


		# embedded = self.embedding(input).view(1, 1, -1)
		# output = embedded
		# output, hidden = self.gru(output, hidden)
		# return output, hidden

	# def initHidden(self):
	# 	return torch.zeros(1, 1, self.hidden_size, device=device)
