import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from matplotlib import pyplot as plt

class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # 初始化hidden和memory cell参数
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, (h_n, h_c) = self.lstm(x, (h0, c0))

        # 选取最后一个时刻的输出
        out = self.fc(out[:, :, :])
        return out

class DCNet(nn.Module):

    def __init__(self, hidden_dim, layer1_dim, layer2_dim):
        super(DCNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(5, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, layer1_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(layer1_dim, layer2_dim)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(layer2_dim, 5)
        self.hidden_init_values = None
        self.hidden = self.init_hidden()
        nn.init.xavier_uniform(self.linear1.weight)
        nn.init.xavier_uniform(self.linear2.weight)
        nn.init.xavier_uniform(self.linear3.weight)

    def init_hidden(self):
        if self.hidden_init_values == None:
            self.hidden_init_values = (autograd.Variable(torch.randn(1, 1, self.hidden_dim)),
                                       autograd.Variable(torch.randn(1, 1, self.hidden_dim)))
        return self.hidden_init_values

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        tmp1 = self.relu1(self.linear1(lstm_out.view(len(seq), -1)))
        tmp2 = self.relu2(self.linear2(tmp1))
        _out = self.linear3(tmp2)
        base_out = _out
        print("_out.shape: ", _out.shape)
        return base_out

import numpy as np

with open("predata.txt", "r") as my_train_file:
    data_1 = my_train_file.read().split(',')

print("data_1: ", data_1[0])

with open("EncodedStrands.txt", "r") as my_train_file:
    data_2 = my_train_file.read().splitlines()

print("data_2: ", data_2[0])



import random
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

bmap = {"A":0, "C":1, "G":2, "T":3, "1": 4}
def one_hot(b):
    t = [[0,0,0,0,0]]
    i = bmap[b]
    t[0][i] = 1
    return t

print("one-hot encoding for DNA bases")
print("A:", one_hot("A"))
print("C:", one_hot("C"))
print("G:", one_hot("G"))
print("T:", one_hot("T"))

seq = [random.choice(["A","C","G","T"]) for _ in range(220)]

seqs = data_1
seqs_2 = data_2

from random import shuffle

c = list(zip(seqs, seqs_2))

shuffle(c)

seqs, seqs_2 = zip(*c)


#print("".join(seq))
# convert the `seq` to a PyTorch tensor

# seq_t = Variable(torch.FloatTensor([one_hot(c) for c in seq])).cuda()
seq_t = torch.Tensor([[one_hot(c) for c in ee] for ee in seqs_2[:20000]])

# seqs_t = [Variable(torch.FloatTensor([one_hot(c) for c in s])).cuda()  for s in seqs]
seqs_t = torch.Tensor([[one_hot(c) for c in e] for e in seqs[:20000]])

# Hyper Parameters
epochs = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
batch_size = 64
time_step = 120      # rnn 时间步数 / 图片高度
input_size = 5     # rnn 每步输入值 / 图片每行像素
hidden_size = 64
num_layers = 1
num_classes = 5
lr = 0.01           # learning rate

dcnet = simpleLSTM(input_size, hidden_size, num_layers, num_classes)

# dcnet = DCNet(32, 12, 12)
# dcnet.cuda()

#loss_function = nn.L1Loss()
loss_function = nn.MSELoss()
# loss_function = nn.CrossEntropyLoss()

# lr = 0.1
optimizer = optim.SGD(dcnet.parameters(), lr=lr)

# ptimizer = torch.optim.Adam(dcnet.parameters(), lr)

range_ = (1, 100)
mini_batch_size = batch_size
for epoch in range(60):
    print("epoch: ", epoch)
    correct = 0
    num = 0
    for i in range(int(len(seqs_t)/mini_batch_size)):
        print("i: ", i)

        images = seqs_t[i * mini_batch_size : (i+1) * mini_batch_size]
        images = torch.mean(images.view(-1, 10, 5), dim=1)

        print("images.shape: ", images.shape)

        images = images.view(-1, time_step, input_size).to(device)
        print("images.shape: ", images.shape)

        labels = seq_t[i * mini_batch_size : (i+1) * mini_batch_size].view(-1, 120, 5)


        labels = labels.to(device)

        # forward pass
        outputs = dcnet(images)

        print("outputs: ", outputs.shape)
        print("labels: ", labels.shape)

        loss = loss_function(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = correct + torch.sum(F.one_hot(torch.argmax(outputs, dim=-1), 5) * labels)

        print("correct: ", correct)

        num = num + 120 * 64

        if(i%1 == 0):
            print("a: ", correct/num)
