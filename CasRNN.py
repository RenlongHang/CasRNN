
import os
import numpy as np
import random

import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA
import skimage.filters.rank as sfr
from skimage.morphology import disk

# dividing the bands into groups by averaging
# when timestep is 3, generate 3 groups
# using hidden outputs of the first LSTM layer as the input of the second LSTM layer

# data path loading
DataPath = '.../IndinePines/Indian_pines_corrected.mat'
TRPath = '.../IndinePines/TRLabel.mat'
TSPath = '.../IndinePines/TSLabel.mat'
savepath = '..../IP-TwoLayerGRU.mat'

batchsize = 64   
LR = 0.001        
EPOCH = 300
HiddenSize1 = 128  
HiddenSize2 = 256  
LstmLayers = 1
TimeStep = 4    

# load data
Data = io.loadmat(DataPath)
TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)

Data = Data['indian_pines_corrected']
Data = Data.astype(np.float32)
TrLabel = TrLabel['TRLabel']
TsLabel = TsLabel['TSLabel']

# normalization method: map to zero mean and one std
[m, n, l] = np.shape(Data)
for i in range(l):
    mean = np.mean(Data[:, :, i])
    std = np.std(Data[:, :, i])
    Data[:, :, i] = (Data[:, :, i] - mean)/std

print np.mean(Data[:, :, 100]), np.std(Data[:, :, 100])

# transform data to matrix
TotalData = np.reshape(Data, [m*n, l])
TrainDataLabel = np.reshape(TrLabel, [m*n, 1])
Tr_index, _ = np.where(TrainDataLabel != 0)
TrainData1 = TotalData[Tr_index, :]
TrainDataLabel = TrainDataLabel[Tr_index, 0]
TestDataLabel = np.reshape(TsLabel, [m*n, 1])
Ts_index, _ = np.where(TestDataLabel != 0)
TestData1 = TotalData[Ts_index, :]
TestDataLabel = TestDataLabel[Ts_index, 0]

print TrainData1.shape, TestData1.shape
# construct data for network
numb = l/TimeStep
TrainData = np.empty((len(TrainDataLabel), numb, TimeStep), dtype='float32')
TestData = np.empty((len(TestDataLabel), numb, TimeStep), dtype='float32')

for i in range(len(TrainDataLabel)):
    temp = TrainData1[i, :]
    for j in range(TimeStep):
        temp2 = temp[j*numb:(j+1)*numb]
        TrainData[i, :, j] = temp2

for i in range(len(TestDataLabel)):
    temp = TestData1[i, :]
    for j in range(TimeStep):
        temp2 = temp[j*numb:(j+1)*numb]
        TestData[i, :, j] = temp2

print TrainData.shape, TestData.shape

TrainData = torch.from_numpy(TrainData)
TrainDataLabel = torch.from_numpy(TrainDataLabel)-1
TrainDataLabel = TrainDataLabel.long()
dataset = dataf.TensorDataset(TrainData, TrainDataLabel)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

TestData = torch.from_numpy(TestData)
TestDataLabel = torch.from_numpy(TestDataLabel)-1
TestDataLabel = TestDataLabel.long()

Classes = len(np.unique(TrainDataLabel))

# # move data to GPU
# TestData = TestData.cuda()
# TestDataLabel = TestDataLabel.cuda()

# construct the network
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        # the first LSTM layer to reduce redundancy
        self.LSTM1 = nn.GRU(
            input_size=1,
            hidden_size=HiddenSize1,
            num_layers=LstmLayers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        # the second LSTM layer to learn complementary
        self.LSTM2 = nn.GRU(
            input_size=HiddenSize1,
            hidden_size=HiddenSize2,
            num_layers=LstmLayers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        # self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(HiddenSize2, Classes)

        self.bn = nn.BatchNorm1d(HiddenSize1, momentum=0.1)


    def forward(self, x):
        input = []

        for i in range(TimeStep):
            temp1 = x[:, :, i].unsqueeze(2)
            r_out1, _ = self.LSTM1(temp1, None)
            temp2 = r_out1[:, -1, :]
            out1 = temp2.unsqueeze(1)
            input.append(out1)

        input2 = input[0]
        for i in range(1, TimeStep):
            temp2 = input[i]
            input2 = torch.cat((input2, temp2), 1)

        out4, _ = self.LSTM2(input2, None)
        output = self.output(out4[:, -1, :])
        return output


lstm = GRU()
lstm.cuda()
print(lstm)

optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data

        b_x = b_x.cuda()
        b_y = b_y.cuda()

        output = lstm(b_x)  # rnn output

        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:

            # lstm.eval()

            # divide test set into many subsets to avoid out-of-memory problem
            pred_y = np.empty((len(TestDataLabel)), dtype='float32')
            number = len(TestDataLabel) / 5000
            for i in range(number):
                temp = TestData[i * 5000:(i + 1) * 5000, :, :]
                temp = temp.cuda()
                temp2 = lstm(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 5000:(i + 1) * 5000] = temp3
                del temp, temp2, temp3

            if (i + 1) * 5000 < len(TestDataLabel):
                temp = TestData[(i + 1) * 5000:len(TestDataLabel), :, :]
                temp = temp.cuda()
                temp2 = lstm(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(TestDataLabel)] = temp3
                del temp, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)
            # test_output = rnn(TestData)
            # pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            # accuracy = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            # lstm.train()

# test each class accuracy
# divide test set into many subsets

# lstm.eval()

pred_y = np.empty((len(TestDataLabel)), dtype='float32')
number = len(TestDataLabel)/5000
for i in range(number):
    temp = TestData[i*5000:(i+1)*5000, :, :]
    temp = temp.cuda()
    temp2 = lstm(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*5000:(i+1)*5000] = temp3
    del temp, temp2, temp3

if (i+1)*5000 < len(TestDataLabel):
    temp = TestData[(i+1)*5000:len(TestDataLabel), :, :]
    temp = temp.cuda()
    temp2 = lstm(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*5000:len(TestDataLabel)] = temp3
    del temp, temp2, temp3

pred_y = torch.from_numpy(pred_y).long()
OA = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)

Classes = np.unique(TestDataLabel)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestDataLabel)):
        if TestDataLabel[j] == cla:
            sum += 1
        if TestDataLabel[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()


print OA
print EachAcc
del TestData, TrainData, TrainDataLabel, b_x, b_y, dataset, train_loader

