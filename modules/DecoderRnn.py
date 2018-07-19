import torch
import torch.nn as nn
from attention import Attention

class DecoderRnn(nn.Module):
	"""docstring for DecoderRnn"""
	def __init__(self, num_layers, hidden_size, embedding=None,
				rnn_type='GRU', bidirectional=False, attn_type="general",
				dropout=0.0,
		 		coverage_attn=False, context_gate=None, reuse_copy_attn=False):

                 
    	super(RNNDecoderBase, self).__init__()

    	self.decoder_type = 'rnn'
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)


        #attention
        self.attention = Attention(attn_type)

        # self.rnn = ####
        self.rnn = getattr(nn, rnn_type)(input_size=embeddings.embedding_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        dropout=dropout,
                                        bidirectional=bidirectional)


        # self.context_gate = None
        # if context_gate is not None:
        #     self.context_gate = onmt.modules.context_gate_factory(
        #         context_gate, self._input_size,
        #         hidden_size, hidden_size, hidden_size
        #     )

        # # Set up the standard attention.
        # self._coverage = coverage_attn
        # self.attn = onmt.modules.GlobalAttention(
        #     hidden_size, coverage=coverage_attn,
        #     attn_type=attn_type
        # )

        # # Set up a separated copy attention layer, if needed.
        # self._copy = False
        # if copy_attn and not reuse_copy_attn:
        #     self.copy_attn = onmt.modules.GlobalAttention(
        #         hidden_size, attn_type=attn_type
        #     )
        # if copy_attn:
        #     self._copy = True
        # self._reuse_copy_attn = reuse_copy_attn

    def forward_step(self, feed, hidden):
        # assert feed.size()[0] == 1
        assert len(feed.size()) == 3
        output,state = self.rnn(feed,hidden)
        return output,state

    def forward(self, memory, target):
        # embedding
        embedded = self.embedding(target)

        # compute hidden state for decoder
        state = self.initHidden()
        # 
        results = []
        attentions = []
        for i,tgt in enumerate(embedded.split()):
            #Teacher forcing
            attn = self.attention(state,memory)
            context = self.compute_context(attn, memory)
            feed = self.compute_feed(tgt,context)
            # feed = feed.unsqueeze(0)
            output, state = self.forward_step(feed, state)
            # output = output.squeeze()
            results.append(output)
            attentions.append(attn)
            #without teacher forcing needed to be implemented

        results = torch.stack(results)
        attentions = torch.stack(attentions)

        return
    
    def compute_feed(self,tgt,context):
        return torch.cat((tgt,context),1)
    
    def compute_context(self, attn, memory):
        attn_ = attn.unsqueeze(-1).expand(-1,-1,memory.size(-1))
        zeros = torch.zeros_like(attn_)
        context = torch.addcmul(zeros,1,attn_,memory).sum(dim=0)
        return context

    def _build_rnn(self):
        return        

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


