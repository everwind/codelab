import sys
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from six.moves import xrange
from torch.nn.parallel.data_parallel import data_parallel
import numpy as np

from lstmtag import LstmTag
from collections import Counter, defaultdict
max_len = 20
criterion = nn.NLLLoss()
class Vocab:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
    def build_vocab(self, infile):
        self.word2id["UNKOWN"] = 0
        for line in open(infile):
            line = "BEGIN_1 "+line+" END_1"
            tokens = line.split()
            for s in tokens:
                ss=s.split("_")
                if ss[0] not in self.word2id:
                    self.word2id[ss[0]] = len(self.word2id)
        self.id2word = {i:w for w,i in self.word2id.iteritems()}
    def id(self, word):
        if word not in self.word2id:
            return 0
        return self.word2id[word]
    def size(self):
        return len(self.word2id)

def read_data(infile,vocab):
    dataset=[]
    targetset=[]
    for i in range(max_len):
        dataset.append([])
        targetset.append([])
    for line in open(infile):
        line = "BEGIN_1 "+line+" END_1"
        items = line.split()
        if len(items) > max_len: continue
        tokens=[]
        tags=[]
        flag = 0
        for s in items:
            ss=s.split("_")
            try:
                tokens.append(vocab.id(ss[0]))
                tags.append(int(ss[1]))
            except:
                flag = 1
        if flag == 1: continue
        dataset[len(tokens)].append(tokens)
        targetset[len(tokens)].append(tags)
    return dataset, targetset

def get_batch(data_set, tag_set, batch_size, batch_first = False):
    rand0 = random.randint(1,max_len-1)
    if len(data_set[rand0]) == 0:
        for i in range(max_len):
            if len(data_set[i]) >0:
                rand0=i
                break
    sents = []
    tags = []
    for i in range(batch_size):
        rand1 = random.randint(0, len(data_set[rand0])-1)
        sents.append(data_set[rand0][rand1])
        tags.append(tag_set[rand0][rand1])

    s=Variable(torch.LongTensor(sents))
    t=Variable(torch.LongTensor(tags))
    if batch_first:
        return s,t
    else:
        return s.t(), t.t()


def evaluate(model, dev_data, dev_tag, batch_size, batch_first=False):
    total_loss = 0
    cnt = 0 
    for _ in xrange(20):
        inputs, tags = get_batch(dev_data, dev_tag, batch_size, batch_first=batch_first)
        if len(inputs) == 0: continue
        #print inputs
        if torch.cuda.is_available():
            inputs, tags = inputs.cuda(), tags.cuda()
        pred = model(inputs)
        if batch_first:
            input_len = inputs.size()[1]
            cnt += inputs.size()[0]
        else:
            input_len = inputs.size()[0]
            cnt += inputs.size()[1]
        for time in xrange(input_len):
            if batch_first:
                y_pred = pred[:, time]
            else:
                y_pred = pred[time]
            if batch_first:
                target = tags[:, time]
            else:
                target = tags[time]
            loss = criterion(y_pred, target)
            total_loss += loss.data
    return total_loss[0] / cnt

def main():
    v1=Vocab()
    v1.build_vocab(sys.argv[1])
    train_data, train_tag = read_data(sys.argv[1],v1)
    dev_data, dev_tag = read_data(sys.argv[2],v1)
    ntag=2
    batch_size = 64 
    init_range = 0.08
    step_per_epoch = 500
    learning_rate = 0.01
    learning_rate_decay = 0.98
    decay_rate = 0.95
    batch_first = True
    checkpoint_after = 500
    device_num = 2
    model =LstmTag(v1.size(), 100, 100, max_len, batch_size, ntag, 3, batch_first=batch_first)

    model_path = "model.dat"
    if os.path.exists(model_path):
        saved_state = torch.load(model_path)
        model.load_state_dict(saved_state)
    else:
        if torch.cuda.is_available():
            model.cuda()
        model.init_weights(init_range)
        optimizer = optim.RMSprop(model.parameters(), lr = learning_rate, alpha=decay_rate)

        train_loss = 0
        last_train_loss = 10
        loss_count = 0
        best_dev_loss = 10
        step = 0
        begin_time = time.time()
        while True:
            inputs, tags = get_batch(train_data, train_tag, batch_size*device_num, batch_first)
            if len(inputs) == 0: continue
            if torch.cuda.is_available():
                inputs, tags = inputs.cuda(), tags.cuda()
            #pred = model(inputs)
            pred = data_parallel(model, inputs,  device_ids=[0, 2])
            total_loss = None
            if batch_first:
                input_len = inputs.size()[1]
            else:
                input_len = inputs.size()[0]
            for time_step in xrange(input_len ):
                if batch_first:
                    y_pred = pred[:,time_step]
                else:
                    y_pred = pred[time_step]
                if batch_first:
                    target = tags[:, time_step]
                else:
                    target = tags[time_step]
                # print(y_pred.size(), target.size())
                loss = criterion(y_pred, target)
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            optimizer.zero_grad()
            total_loss /= batch_size
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss

            if step % step_per_epoch == 0:
                epoch = step / step_per_epoch
                dev_loss = evaluate(model, dev_data, dev_tag, batch_size, batch_first)
                train_loss = train_loss.data[0] / step_per_epoch
                if train_loss > last_train_loss:
                    loss_count += 1
                    if loss_count == 3:
                        learning_rate *= learning_rate_decay
                        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=decay_rate)
                        loss_count = 0
                else:
                    loss_count = 0
                last_train_loss = train_loss
                if epoch > 0:
                    epoch_time = (time.time() - begin_time) / epoch
                else:
                    epoch_time = 0
                print("Epoch time: {0}\tEpoch: {1}\tLR: {2}, Train loss: {3}\tDev loss: {4}".format(
                    epoch_time, epoch, learning_rate, train_loss, dev_loss
                    ))
                train_loss = 0
                if epoch > checkpoint_after and dev_loss < best_dev_loss:
                    state_to_save = model.state_dict()
                    torch.save(state_to_save, model_path)

            step += 1

main()
