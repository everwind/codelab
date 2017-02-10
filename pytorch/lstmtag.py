from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

# Took from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
def repackage_state(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_state(v) for v in h)


class LstmTag(nn.Module):
    def __init__(self, input_ntoken, emb_size, hidden_size,
            input_max_len, batch_size, ntag,
            nlayers=1, bias=False, dropout_p=0.5,
            batch_first=False):
        super(LstmTag, self).__init__()

        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

        # encoder stack
        #self.linear = nn.Linear(hidden_size*2, ntag, bias=True)
        self.embedding = nn.Embedding(input_ntoken, emb_size)
        self.encoder = nn.LSTM(emb_size, hidden_size, nlayers, bias=bias, batch_first=False)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv2d(1, ntag, (1,hidden_size*2))
        self.softmax = nn.LogSoftmax()

       
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nlayers = nlayers 
        self.ntag = ntag       
        self.input_max_len = input_max_len       
        self.batch_first = batch_first

    def init_weights(self, initrange):
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def predict(self, encoder_inputs):
        if self.batch_first :
            encoder_inputs = encoder_inputs.transpose(0,1)
        weight = next(self.parameters()).data
        init_state = (Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_()))
        embedding = self.embedding(encoder_inputs)
        input_len = encoder_inputs.size()[0]
        indice = Variable(torch.linspace(input_len-1,0,input_len).long())
        if encoder_inputs.is_cuda:
            indice = indice.cuda()
        reverse_inputs = torch.index_select(encoder_inputs, 0, indice)
        reverse_embeding = self.embedding(reverse_inputs)
        if self.dropout_p > 0:
            embedding = self.dropout(embedding)
        encoder_outputs, encoder_state = self.encoder(embedding, init_state)
        reverse_outputs, reverse_state = self.encoder(reverse_embeding, init_state)
        bi_outputs = torch.cat((encoder_outputs, encoder_outputs),2)
        # To calculate W1 * bi_outputs[i] , we use a 1-by-1 convolution, need to reshape before.
        sz = bi_outputs.size()
        y = self.conv(bi_outputs.view(sz[0], 1, sz[1], sz[2])) #  # input_len * hidden_len * batch_len 
        y= y.squeeze().transpose(1,2)  # input_len * batch_len * hidden_len
        
        out=[]
        for i in range(y.size()[0]): 
            out.append(self.softmax(y[i]))

        if self.batch_first:
            return torch.stack(out,1)
        else:
            return torch.stack(out)


    def forward(self, encoder_inputs):
        # encoding
        pred = self.predict(encoder_inputs)
        return pred


