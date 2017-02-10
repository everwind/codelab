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


class Seq2Seq(nn.Module):
    def __init__(self, encode_ntoken, decode_ntoken,
            input_size, hidden_size,
            input_max_len, output_max_len,
            batch_size,
            nlayers=1, bias=False, attention=True, dropout_p=0.5,
            batch_first=False):
        super(Seq2Seq, self).__init__()
        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

        # encoder stack
        self.enc_embedding = nn.Embedding(encode_ntoken, input_size)
        self.encoder = nn.LSTM(input_size, hidden_size, nlayers, bias=bias, batch_first=batch_first)

        # decoder stack
        self.dec_embedding = nn.Embedding(decode_ntoken, input_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, nlayers, bias=bias, batch_first=batch_first)
        if attention:
            self.attn_enc_linear = nn.Linear(hidden_size, hidden_size)
            self.attn_dec_linear = nn.Linear(hidden_size, hidden_size)
            self.attn_linear = nn.Linear(hidden_size*2, hidden_size)
            self.att_weight = Parameter(torch.Tensor(hidden_size, 1))
            self.attn_tanh = nn.Tanh()
            self.attn_conv = nn.Conv2d(1, hidden_size, (1,hidden_size))
        self.linear = nn.Linear(hidden_size, decode_ntoken, bias=True)

        self.softmax = nn.LogSoftmax()

        self.decode_ntoken = decode_ntoken
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.attention = attention
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len
        self.batch_first = batch_first

    def init_weights(self, initrange):
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def attention_func(self, encoder_outputs, decoder_hidden):
        #calculate U*h_j
        att_dec_tmp = self.attn_dec_linear(decoder_hidden)
        sz = encoder_outputs.size()
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        attn_value = self.attn_conv(encoder_outputs.view(sz[0], 1, sz[1], sz[2]))
        attn_value = attn_value.squeeze().transpose(1,2)

        ss = torch.stack([att_dec_tmp]*sz[0])
        attn_value = self.attn_tanh(attn_value + ss)

        # input_len * batch * 1 -->  batch * input_len * 1
        attn_value2 = torch.bmm(attn_value, self.att_weight.expand(self.hidden_size, self.input_max_len).t().unsqueeze(2))
        attention = nn.Softmax()(torch.transpose(attn_value2.squeeze(),0,1))

        hidden = torch.bmm(torch.transpose(torch.transpose(encoder_outputs, 0, 1), 1, 2), attention.unsqueeze(2))       
        return hidden.squeeze()

    def attention_func1(self, encoder_outputs, decoder_hidden):
        #att_enc_tmp = Variable(torch.Tensor(self.input_size, self.batch_size, self.hidden_size))
        att_dec_tmp = self.attn_dec_linear(decoder_hidden)
        attn_value = Variable(encoder_outputs.data.new(self.input_max_len, self.batch_size, self.hidden_size))
        for i in range(encoder_outputs.size()[0]):
            attn_value[i] = self.attn_tanh(self.attn_enc_linear(encoder_outputs[i]) + att_dec_tmp)

        # input_len * batch * 1 -->  batch * input_len * 1
        attn_value2 = torch.bmm(attn_value, self.att_weight.expand(self.hidden_size, self.input_max_len).t().unsqueeze(2))
        attention = nn.Softmax()(torch.transpose(attn_value2.squeeze(),0,1))

        #hidden = encoder_outputs
        #for i in range(1, att_enc_tmp.size()[0]):
        #    hidden = hidden + att_enc_tmp[i].attention[i]
        #    (batch_size * hidden_len * input_len ) * (batch_size * input_len) = batch_size * hidden_len

        hidden = torch.bmm(torch.transpose(torch.transpose(encoder_outputs, 0, 1), 1, 2), attention.unsqueeze(2))       
        return hidden.squeeze()



    def encode(self, encoder_inputs):
        weight = next(self.parameters()).data
        init_state = (Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_()))
        embedding = self.enc_embedding(encoder_inputs)
        if self.dropout_p > 0:
            embedding = self.dropout(embedding)
        encoder_outputs, encoder_state = self.encoder(embedding, init_state)
        return encoder_outputs, encoder_state

    def decode(self, encoder_outputs, encoder_state, decoder_inputs, feed_previous):
        pred = []
        state = encoder_state
        if feed_previous:
            if self.batch_first:
                embedding = self.dec_embedding(decoder_inputs[:,0].unsqueeze(1))
            else:
                embedding = self.dec_embedding(decoder_inputs[0].unsqueeze(0))
            for time in range(1, self.output_max_len):
                #state = repackage_state(state)
                # batch_size * 1 * embedding_size
                output, state = self.decoder(embedding, state)
                # print(output.size())

                att_out = self.attention_func(encoder_outputs, output.squeeze())
                softmax = self.predict(output.squeeze(), att_out)
                # feed previous
                decoder_input = softmax.max(1)[1]
                embedding = self.dec_embedding(decoder_input.squeeze().unsqueeze(0))
                if self.batch_first:
                    embedding = torch.transpose(embedding, 0, 1)
                pred.append(softmax)
        else:
            embedding = self.dec_embedding(decoder_inputs)
            if self.dropout_p > 0:
                embedding = self.dropout(embedding)
            outputs, _ = self.decoder(embedding, state)
            # print(outputs.size())

            # if self.batch_first:
                # for batch in range(self.batch_size):
                    # output = outputs[batch,1:,:]
                    # softmax = self.predict(output, encoder_outputs)
                    # pred.append(softmax)
            # else:
            for time_step in range(self.output_max_len - 1):
                if self.batch_first:
                    output = outputs[:,time_step,:]
                else:
                    output = outputs[time_step]
                #softmax = self.predict(output, encoder_outputs)
                att_out = self.attention_func(encoder_outputs, output)
                softmax = self.predict(output, att_out)
                pred.append(softmax)
        return pred

    def predict(self, dec_output, att_out):

        if self.dropout_p > 0:
            dec_output = self.dropout(dec_output)
            att_out = self.dropout(att_out)
        x = self.attn_linear(torch.cat((dec_output,att_out),1))
        linear = self.linear(x)
        softmax = self.softmax(linear)
        return softmax

    #def forward(self, inputs, feed_previous=False):
    def forward(self, encoder_inputs, decoder_inputs, feed_previous=False):
        #encoder_inputs = inputs[0]
        #decoder_inputs = inputs[1]
        # encoding
        encoder_outputs, encoder_state = self.encode(encoder_inputs)

        # decoding
        pred = self.decode(encoder_outputs, encoder_state, decoder_inputs, feed_previous)
        return pred

class Seq2Tree(Seq2Seq):
    def decode(self, encoder_outputs, encoder_state, decoder_inputs, feed_previous):
        pass

    def forward(self, encoder_inputs, decoder_tree, feed_previous=False):
        # encoding
        encoder_outputs, encoder_state = self.encode(encoder_inputs)

        # decoding
        pred = self.decode(encoder_outputs, encoder_state, decoder_tree, feed_previous)
        return pred
