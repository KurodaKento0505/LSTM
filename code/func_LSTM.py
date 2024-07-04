'''ライブラリの準備'''
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.utils.rnn as rnn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import optuna
import csv
import pandas as pd
import os
import random
import numpy as np


'''GPUチェック'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def LSTM(sequence_np, label_np, num_tactical_action_per_training, tactical_action_name, dim_of_image, train_truth, make_graph, val): 


    ##################################################################
    batch_size = 2048
    hidden_dim = 20
    epoch = 500
    lr = 0.001
    ##################################################################


    # slice する数を計算
    slice_len = len(sequence_np) % 10
    sequence_np = np.delete(sequence_np, slice(len(sequence_np) - slice_len, len(sequence_np)), 0)
    label_np = np.delete(label_np, slice(len(sequence_np) - slice_len, len(sequence_np)), 0)


    # torch.tensorでtensor型に
    train_x = torch.from_numpy(sequence_np.astype(np.float32)).clone()
    train_t = torch.from_numpy(label_np.astype(np.float32)).clone()

    print('train_x:', train_x.shape)
    print('train_t:', train_t.shape)


    if train_truth:
        dataset = torch.utils.data.TensorDataset(train_x, train_t)

        train_size = int(len(dataset) * 0.8) # train_size is 3000
        val_size = int(len(dataset) * 0.1) # val_size is 1000
        test_size = int(len(dataset) * 0.1)# val_size is 1000
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(3)) # 42
        
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = True, num_workers = 0)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 0)


    else:
        if make_graph:
            graph_test_dataset = torch.utils.data.TensorDataset(train_x, train_t)
            graph_testloader = torch.utils.data.DataLoader(graph_test_dataset, batch_size, shuffle = False, num_workers = 0)

        else:
            dataset = torch.utils.data.TensorDataset(train_x, train_t)
            train_size = int(len(dataset) * 0.6) # train_size is 3000
            val_size = int(len(dataset) * 0.2) # val_size is 1000
            test_size = int(len(dataset) * 0.2)# val_size is 1000
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(3)) # 42
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0)
            valloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = True, num_workers = 0)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 0)


    # 時間を入れるか入れないか input_dim=3 or 2
    # only_x : * 1 + 2, x + y : * 2 + 3
    model = LSTMClassification(input_dim = dim_of_image, 
                            hidden_dim = hidden_dim, 
                            target_size = num_tactical_action_per_training)


    PATH = './tisc_output/LSTM/train_model/' + tactical_action_name + '_net.pth'

    if train_truth:
        train(model, epoch, trainloader, valloader, lr)
        torch.save(model.state_dict(), PATH)
        evaluate(model, testloader, train_truth)

    else:
        model.load_state_dict(torch.load(PATH), strict=False)

        if make_graph:
            outputs_list, labels_list, len_loader = evaluate(model, graph_testloader, num_tactical_action_per_training)
        else:
            if val:
                outputs_list, labels_list, len_loader = evaluate(model, valloader, num_tactical_action_per_training)
            else:
                outputs_list, labels_list, len_loader = evaluate(model, testloader, num_tactical_action_per_training)
            
        return outputs_list, labels_list, len_loader


    # val_accuracy = evaluate(model, valloader)
    # test_accuracy = evaluate(model, testloader)
    # print("Test Accuracy: {:.2f}%".format(val_accuracy * 100))
    # print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))



# モデル
class LSTMClassification(nn.Module):

        def __init__(self, input_dim, hidden_dim, target_size):
            super(LSTMClassification, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, target_size)

        # 実際に動かす
        def forward(self, input_):
            lstm_out, (h, c) = self.lstm(input_)
            logits = self.fc(lstm_out[:,-1])
            scores = logits
            # scores = torch.sigmoid(logits)
            # print(np.shape(scores))
            return scores



def train(model, n_epochs, trainloader, valloader, lr):
    model = model.to(device)
    loss_function = nn.HuberLoss() # SmoothL1, CrossEntropyLoss, HuberLoss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = {
        'loss': []
    }
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        val_losses = []

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.long()

            model.zero_grad()

            tag_scores = model(inputs)

            # labels = labels.unsqueeze(1)
            
            train_loss = loss_function(tag_scores, labels.float())
            
            train_loss.backward()
            optimizer.step()
            train_losses.append(float(train_loss))

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()
                model.zero_grad()
                tag_scores = model(inputs)
                val_loss = loss_function(tag_scores, labels.float())
                val_losses.append(float(val_loss))

        avg_train_loss = np.mean(train_losses)
        history['loss'].append(avg_train_loss)
        print("Epoch {} / {}: train_Loss = {:.3f}".format(epoch+1, n_epochs, avg_train_loss))

        avg_val_loss = np.mean(val_losses)
        history['loss'].append(avg_val_loss)
        print("Epoch {} / {}: val_Loss = {:.3f}".format(epoch+1, n_epochs, avg_val_loss))

    return history



def evaluate(model, loader, number_of_tactical_action):
    model = model.to(device)
    model.eval()

    # total = 0

    outputs_list = []
    labels_list = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels_i = data
            inputs,labels_i = inputs.to(device), labels_i.to(device)
            labels_i_list = labels_i.tolist()

            outputs_i = model(inputs)
            # outputs_i = F.softmax(outputs_i, dim = 1)
            outputs_i_list = outputs_i.tolist()

            for j in range(len(outputs_i)):
                outputs_list.append(outputs_i_list[j])

            for j in range(len(labels_i)):
                labels_list.append(labels_i_list[j])

            # total += labels_i.size(0)

        # percent_calculate(outputs_list, labels_list, len(loader), test, number_of_tactical_action, make_graph, val)
    # accuracy = correct / total
    # return accuracy
    return outputs_list, labels_list, len(loader)