'''ライブラリの準備'''
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.utils.rnn as rnn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

'''GPUチェック'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import csv
import pandas as pd
import os
import random
import numpy as np


def main():

    '''labels = "no_counter"

    competition_name = "FIFA_World_Cup_2022"  # FIFA_World_Cup_2022, UEFA_Euro_2020, UEFA_Women's_Euro_2022, Women's_World_Cup_2023

    # ラベル変更かつ全部同じファイルに
    label(labels,competition_name)'''


    # data_length
    data_length = 11

    # number of player
    number_of_player = 0


    dir = "c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\only_ball\\FIFA_World_Cup_2022"
    
    # ファイルの長さ
    file_length = count_file(dir)

    # sequenceの長さ
    # sequence_length(dir, file_length)

    # sequenceのnumpyを作成
    # （ボール + 選手22人）x 2 + 時間 + ラベル = 44
    sequence_np = np.zeros([file_length,data_length, number_of_player * 2 + 3])

    # labelのnumpyを作成
    label_np = np.zeros([file_length])


    # count_player
    # count_player(dir, file_length)
    

    for i in range(file_length):

        df = pd.read_csv(dir + "\\" + str(i + 1).zfill(6) + ".csv")


        # ラベルに応じてtrain_tにラベル付与
        if df.at[0,"sequence_label"] == 1:
            label_np[i] = 1

        elif df.at[0,"sequence_label"] == 2:
            label_np[i] = 2


        # put in segmentstion data from df to sequence_np
        put_in_seg_data(df, sequence_np, i)
        
        # put in attack_sequence data from df to sequence_np
        # put_in_360_data(df, sequence_np, data_length, i)


        # sort_player_position
        # sort_player(df)


        # 360 data 整形
        # indentify_player(sequence_np, number_of_player, data_length, i)


        if i % 1000 == 0:
            print(i)
    
    # 転置
    label_np = label_np.T

    print(sequence_np)
    print(label_np)


    # numpy保存
    np.save('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\data\\FIFA_World_Cup_2022\\sequence_np', sequence_np)
    np.save('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\data\\FIFA_World_Cup_2022\\label_np', label_np)



    '''# numpy load
    sequence_np = np.load('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\data\\FIFA_World_Cup_2022\\sequence_np.npy')
    label_np = np.load('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\data\\FIFA_World_Cup_2022\\label_np.npy')


    # ラベルごとに分割
    # Timeありかなしかで 2 or 3
    sequence_0_np = np.zeros([76098, data_length, number_of_player * 2 + 3], dtype = np.float16)
    sequence_1_np = np.zeros([848, data_length, number_of_player * 2 + 3], dtype = np.float16)
    sequence_2_np = np.zeros([1013, data_length, number_of_player * 2 + 3], dtype = np.float16)

    label_0_np = np.zeros([76098])
    label_1_np = np.zeros([848])
    label_2_np = np.zeros([1013])

    a = 0 # その他の攻撃の数
    b = 0 # ロングカウンターの数
    c = 0 # ショートカウンターの数
    for i in range(len(label_np)):
        if label_np[i] == 0:
            sequence_0_np[a] = sequence_np[i]
            label_0_np[a] = label_np[i]
            a += 1
        elif label_np[i] == 1:
            sequence_1_np[b] = sequence_np[i]
            label_1_np[b] = label_np[i]
            b += 1
        elif label_np[i] == 2:
            sequence_2_np[c] = sequence_np[i]
            label_2_np[c] = label_np[i]
            c += 1'''

    # LSTM(sequence_0_np, label_0_np, sequence_1_np, label_1_np, sequence_2_np, label_2_np, number_of_player)




def LSTM(train_0_x, train_0_t, train_1_x, train_1_t, train_2_x, train_2_t, number_of_player):

    # torch.tensorでtensor型に
    # train_x = 

    train_0_x = torch.from_numpy(train_0_x.astype(np.float32)).clone()
    train_0_t = torch.from_numpy(train_0_t.astype(np.int32)).clone()
    train_1_x = torch.from_numpy(train_1_x.astype(np.float32)).clone()
    train_1_t = torch.from_numpy(train_1_t.astype(np.int32)).clone()
    train_2_x = torch.from_numpy(train_2_x.astype(np.float32)).clone()
    train_2_t = torch.from_numpy(train_2_t.astype(np.int32)).clone()

    '''# ラベル数確認
    a0 = 0
    a1 = 0
    a2 = 0
    for i in range(len(train_t)):
        if train_t[i] == 0:
            a0 += 1
        elif train_t[i] == 1:
            a1 += 1
        elif train_t[i] == 2:
            a2 += 1
    
    print('0:',a0,'1:',a1,'2:',a2)'''

    dataset_0 = torch.utils.data.TensorDataset(train_0_x, train_0_t)
    dataset_1 = torch.utils.data.TensorDataset(train_1_x, train_1_t)
    dataset_2 = torch.utils.data.TensorDataset(train_2_x, train_2_t)


    # trainとvalidationとtestをsplit
    train_size_0 = int(len(dataset_0) * 0.6) # train_size is 3000
    val_size_0 = int(len(dataset_0) * 0.2) # val_size is 1000
    test_size_0 = int(len(dataset_0) * 0.2)# val_size is 1000
    train_dataset_0, val_dataset_0, test_dataset_0 = torch.utils.data.random_split(dataset_0, [train_size_0, val_size_0, test_size_0], torch.Generator().manual_seed(3)) # 42

    train_size_1 = int(len(dataset_1) * 0.6) # train_size is 600
    val_size_1 = int(len(dataset_1) * 0.2) # val_size is 200
    test_size_1 = int(len(dataset_1) * 0.2)# val_size is 200
    train_dataset_1, val_dataset_1, test_dataset_1 = torch.utils.data.random_split(dataset_1, [train_size_1, val_size_1, test_size_1], torch.Generator().manual_seed(3)) # 42

    train_size_2 = int(len(dataset_2) * 0.6) # train_size is 1200
    val_size_2 = int(len(dataset_2) * 0.2) # val_size is 400
    test_size_2 = int(len(dataset_2) * 0.2)# val_size is 400
    train_dataset_2, val_dataset_2, test_dataset_2 = torch.utils.data.random_split(dataset_2, [train_size_2, val_size_2, test_size_2], torch.Generator().manual_seed(3)) # 42


    # 合体
    train_dataset = train_dataset_0 + train_dataset_1 + train_dataset_2
    val_dataset = val_dataset_0 + val_dataset_1 + val_dataset_2
    test_dataset = test_dataset_0 + test_dataset_1 + test_dataset_2


    batch_size = 512
    
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = True, num_workers = 0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 0)
    

    # 時間を入れるか入れないか input_dim=3 or 2
    # only ball : 3, 360_data : 43
    model = LSTMClassification(input_dim = number_of_player * 2 + 3, 
                            hidden_dim = 6, 
                            target_size = 3)
    
    
    epoch = 1000

    train(model, epoch, trainloader)

    '''PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

    model.load_state_dict(torch.load(PATH))'''

    # val_accuracy = evaluate(model, valloader)
    test_accuracy = evaluate(model, testloader)
    # print("Test Accuracy: {:.2f}%".format(val_accuracy * 100))
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))



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
        



def train(model, n_epochs, trainloader):
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    history = {
        'loss': []
    }
    for epoch in range(n_epochs):
        losses = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.long()

            model.zero_grad()

            tag_scores = model(inputs)

            # labels = labels.unsqueeze(1)
            
            loss = loss_function(tag_scores, labels)
            
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        avg_loss = np.mean(losses)
        history['loss'].append(avg_loss)
        print("Epoch {} / {}: Loss = {:.3f}".format(epoch+1, n_epochs, avg_loss))
    return history



def evaluate(model, loader):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    predicted = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels_i = data
            inputs,labels_i = inputs.to(device), labels_i.to(device)
            outputs = model(inputs)
            _, predicted_i = torch.max(outputs, 1) 
            print(labels_i.size(0))
            total += labels_i.size(0)
            correct += (predicted_i == labels_i).sum().item()
            predicted.append(predicted_i)
            labels.append(labels_i)
        predicted = torch.cat(predicted)
        labels = torch.cat(labels)
        print(predicted, labels)
        calculate(predicted,labels)
    accuracy = correct / total
    return accuracy



# modelの評価
def calculate(predicted,labels):

    # cpuに移動
    predicted = torch.Tensor.cpu(predicted)
    labels = torch.Tensor.cpu(labels)

    print(confusion_matrix(labels, predicted))

    print(classification_report(labels, predicted))



# データにラベルをつける ＆ 同じファイルに番号変えて移動
def label(labels,competition_name):

    # ラベル付与
    i = 0
    for i in range(count_file(labels,competition_name)):
        if i >= 2000:
            break
        df = pd.read_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\360_data\\" + str(labels) + "\\" + str(competition_name) + "\\" + str(i + 1).zfill(6) + ".csv")
        
        '''# ラベル付与
        if labels == "no_counter":
            df["label"] = 0
        elif labels == "longcounter":
            df["label"] = 1
        elif labels == "shortcounter":
            df["label"] = 2
        else:
            break'''

        # ファイル移動
        # 前のラベルが何個入ったか
        s = i + 1000
        df.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\360_data\\all\\" + str(s + 1).zfill(6) + ".csv")
        
        print(str(s + 1).zfill(6))
        '''if i % 1000 == 0:
            print(i)'''



# ディレクトリ内のファイル数調査
def count_file(a):# labels,competition_name  or  a  or  competition_name
    # dir = "C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\360_data\\" + str(labels) + "\\" + str(competition_name) 
    dir = a
    count_file = 0
    
    #ディレクトリの中身分ループ
    for file_name in os.listdir(dir):
    
        #ファイルもしくはディレクトリのパスを取得
        file_path = os.path.join(dir,file_name)
    
        #ファイルであるか判定
        if os.path.isfile(file_path):
            count_file +=1
    
    return count_file



def put_in_seg_data(df, sequence_np, i):
    for j in range(len(df)):
        sequence_np[i, j, 0] = df.at[j,'start_x']
        sequence_np[i, j, 1] = df.at[j,'start_y']
        sequence_np[i, j, 2] = df.at[j,'time_seconds'] / 100



# シーケンスの長さをcsvに
def sequence_length(dir, file_length):
    sequence_length_list = []
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    max_len = 0
    for i in range(file_length):
        df = pd.read_csv(dir + "\\" + str(i + 1).zfill(6) + ".csv")
        if max_len < len(df):
            max_len = len(df)
        elif (len(df) >= 1) and (len(df) <= 5):
            a += 1
        elif (len(df) >= 6) and (len(df) <= 10):
            b += 1
        elif (len(df) >= 11) and (len(df) <= 15):
            c += 1
        elif (len(df) >= 16) and (len(df) <= 20):
            d += 1
        elif (len(df) >= 21) and (len(df) <= 30):
            e += 1
        else:
            f += 1
        
        if i % 1000 == 0:
            print(i)

    print(max_len)

    sequence_length_list.append(a)
    sequence_length_list.append(b)
    sequence_length_list.append(c)
    sequence_length_list.append(d)
    sequence_length_list.append(e)
    sequence_length_list.append(f)
    
    csv_path = r"c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\360_data\\sequence_length.csv"

    sequence_length_np = np.array(sequence_length_list)

    np.savetxt(csv_path, sequence_length_np, fmt='%s', delimiter=',')


# put in data from df to sequence_np
def put_in_360_data(df, sequence_np, data_length, i):
    if len(df) <= data_length:
        for j in range(len(df)):
            sequence_np[i, j + data_length - len(df), 0] = df.at[j,'start_x']
            sequence_np[i, j + data_length - len(df), 1] = df.at[j,'start_y']
            sequence_np[i, j + data_length - len(df), 2] = df.at[j,'time_seconds'] / 100
            sequence_np[i, j + data_length - len(df), 3] = df.at[j,'teammate_1_x']
            sequence_np[i, j + data_length - len(df), 4] = df.at[j,'teammate_1_y']
            sequence_np[i, j + data_length - len(df), 5] = df.at[j,'opponent_player_1_x']
            sequence_np[i, j + data_length - len(df), 6] = df.at[j,'opponent_player_1_y']
            sequence_np[i, j + data_length - len(df), 7] = df.at[j,'teammate_2_x']
            sequence_np[i, j + data_length - len(df), 8] = df.at[j,'teammate_2_y']
            sequence_np[i, j + data_length - len(df), 9] = df.at[j,'opponent_player_2_x']
            sequence_np[i, j + data_length - len(df), 10] = df.at[j,'opponent_player_2_y']
            sequence_np[i, j + data_length - len(df), 11] = df.at[j,'teammate_3_x']
            sequence_np[i, j + data_length - len(df), 12] = df.at[j,'teammate_3_y']
            sequence_np[i, j + data_length - len(df), 13] = df.at[j,'opponent_player_3_x']
            sequence_np[i, j + data_length - len(df), 14] = df.at[j,'opponent_player_3_y']
            sequence_np[i, j + data_length - len(df), 15] = df.at[j,'teammate_4_x']
            sequence_np[i, j + data_length - len(df), 16] = df.at[j,'teammate_4_y']
            sequence_np[i, j + data_length - len(df), 17] = df.at[j,'opponent_player_4_x']
            sequence_np[i, j + data_length - len(df), 18] = df.at[j,'opponent_player_4_y']
            sequence_np[i, j + data_length - len(df), 19] = df.at[j,'teammate_5_x']
            sequence_np[i, j + data_length - len(df), 20] = df.at[j,'teammate_5_y']
            sequence_np[i, j + data_length - len(df), 21] = df.at[j,'opponent_player_5_x']
            sequence_np[i, j + data_length - len(df), 22] = df.at[j,'opponent_player_5_y']
            sequence_np[i, j + data_length - len(df), 23] = df.at[j,'teammate_6_x']
            sequence_np[i, j + data_length - len(df), 24] = df.at[j,'teammate_6_y']
            sequence_np[i, j + data_length - len(df), 25] = df.at[j,'opponent_player_6_x']
            sequence_np[i, j + data_length - len(df), 26] = df.at[j,'opponent_player_6_y']
            sequence_np[i, j + data_length - len(df), 27] = df.at[j,'teammate_7_x']
            sequence_np[i, j + data_length - len(df), 28] = df.at[j,'teammate_7_y']
            sequence_np[i, j + data_length - len(df), 29] = df.at[j,'opponent_player_7_x']
            sequence_np[i, j + data_length - len(df), 30] = df.at[j,'opponent_player_7_y']
            sequence_np[i, j + data_length - len(df), 31] = df.at[j,'teammate_8_x']
            sequence_np[i, j + data_length - len(df), 32] = df.at[j,'teammate_8_y']
            sequence_np[i, j + data_length - len(df), 33] = df.at[j,'opponent_player_8_x']
            sequence_np[i, j + data_length - len(df), 34] = df.at[j,'opponent_player_8_y']
            sequence_np[i, j + data_length - len(df), 35] = df.at[j,'teammate_9_x']
            sequence_np[i, j + data_length - len(df), 36] = df.at[j,'teammate_9_y']
            sequence_np[i, j + data_length - len(df), 37] = df.at[j,'opponent_player_9_x']
            sequence_np[i, j + data_length - len(df), 38] = df.at[j,'opponent_player_9_y']
            sequence_np[i, j + data_length - len(df), 39] = df.at[j,'teammate_10_x']
            sequence_np[i, j + data_length - len(df), 40] = df.at[j,'teammate_10_y']
            sequence_np[i, j + data_length - len(df), 41] = df.at[j,'opponent_player_10_x']
            sequence_np[i, j + data_length - len(df), 42] = df.at[j,'opponent_player_10_y']
            sequence_np[i, j + data_length - len(df), 43] = df.at[j,'teammate_11_x']
            sequence_np[i, j + data_length - len(df), 44] = df.at[j,'teammate_11_y']
            sequence_np[i, j + data_length - len(df), 45] = df.at[j,'opponent_player_11_x']
            sequence_np[i, j + data_length - len(df), 46] = df.at[j,'opponent_player_11_y']
    else:
        for j in range(data_length):
            # len(df) > data_length 頭を切るか、終わりを切るか j or j + len(df) - data_length
            sequence_np[i,j,0] = df.at[j,'start_x']
            sequence_np[i,j,1] = df.at[j,'start_y']
            sequence_np[i,j,2] = df.at[j,'time_seconds'] / 100
            sequence_np[i, j, 3] = df.at[j,'teammate_1_x']
            sequence_np[i, j, 4] = df.at[j,'teammate_1_y']
            sequence_np[i, j, 5] = df.at[j,'opponent_player_1_x']
            sequence_np[i, j, 6] = df.at[j,'opponent_player_1_y']
            sequence_np[i, j, 7] = df.at[j,'teammate_2_x']
            sequence_np[i, j, 8] = df.at[j,'teammate_2_y']
            sequence_np[i, j, 9] = df.at[j,'opponent_player_2_x']
            sequence_np[i, j, 10] = df.at[j,'opponent_player_2_y']
            sequence_np[i, j, 11] = df.at[j,'teammate_3_x']
            sequence_np[i, j, 12] = df.at[j,'teammate_3_y']
            sequence_np[i, j, 13] = df.at[j,'opponent_player_3_x']
            sequence_np[i, j, 14] = df.at[j,'opponent_player_3_y']
            sequence_np[i, j, 15] = df.at[j,'teammate_4_x']
            sequence_np[i, j, 16] = df.at[j,'teammate_4_y']
            sequence_np[i, j, 17] = df.at[j,'opponent_player_4_x']
            sequence_np[i, j, 18] = df.at[j,'opponent_player_4_y']
            sequence_np[i, j, 19] = df.at[j,'teammate_5_x']
            sequence_np[i, j, 20] = df.at[j,'teammate_5_y']
            sequence_np[i, j, 21] = df.at[j,'opponent_player_5_x']
            sequence_np[i, j, 22] = df.at[j,'opponent_player_5_y']
            sequence_np[i, j, 23] = df.at[j,'teammate_6_x']
            sequence_np[i, j, 24] = df.at[j,'teammate_6_y']
            sequence_np[i, j, 25] = df.at[j,'opponent_player_6_x']
            sequence_np[i, j, 26] = df.at[j,'opponent_player_6_y']
            sequence_np[i, j, 27] = df.at[j,'teammate_7_x']
            sequence_np[i, j, 28] = df.at[j,'teammate_7_y']
            sequence_np[i, j, 29] = df.at[j,'opponent_player_7_x']
            sequence_np[i, j, 30] = df.at[j,'opponent_player_7_y']
            sequence_np[i, j, 31] = df.at[j,'teammate_8_x']
            sequence_np[i, j, 32] = df.at[j,'teammate_8_y']
            sequence_np[i, j, 33] = df.at[j,'opponent_player_8_x']
            sequence_np[i, j, 34] = df.at[j,'opponent_player_8_y']
            sequence_np[i, j, 35] = df.at[j,'teammate_9_x']
            sequence_np[i, j, 36] = df.at[j,'teammate_9_y']
            sequence_np[i, j, 37] = df.at[j,'opponent_player_9_x']
            sequence_np[i, j, 38] = df.at[j,'opponent_player_9_y']
            sequence_np[i, j, 39] = df.at[j,'teammate_10_x']
            sequence_np[i, j, 40] = df.at[j,'teammate_10_y']
            sequence_np[i, j, 41] = df.at[j,'opponent_player_10_x']
            sequence_np[i, j, 42] = df.at[j,'opponent_player_10_y']
            sequence_np[i, j, 43] = df.at[j,'teammate_11_x']
            sequence_np[i, j, 44] = df.at[j,'teammate_11_y']
            sequence_np[i, j, 45] = df.at[j,'opponent_player_11_x']
            sequence_np[i, j, 46] = df.at[j,'opponent_player_11_y']
    
    return sequence_np


# identify player
def indentify_player(sequence_np, number_of_player, data_length, i):
    for j in range(data_length):
        for k in range(3, number_of_player * 2 + 3):
            
            # 最初はキャンセル
            if j == 0:
                continue

            # ｰ1.0が入っているデータはキャンセル
            if k % 2 == 1:
                if (sequence_np[i, j, k] == -1.0).any().any() or (sequence_np[i, j - 1, k] == -1.0).any().any() or (sequence_np[i, j, k + 1] == -1.0).any().any() or (sequence_np[i, j - 1, k + 1] == -1.0).any().any():
                    continue

                # 前のデータと30以上離れていたら違う選手の可能性
                limit_difference = 15
                if (sequence_np[i, j, k] - sequence_np[i, j - 1, k] >= limit_difference).any().any() and (sequence_np[i, j, k + 1] - sequence_np[i, j - 1, k + 1] >= limit_difference).any().any():
                    print(sequence_np[i, j, k], sequence_np[i, j - 1, k], sequence_np[i, j, k + 1], sequence_np[i, j - 1, k + 1])

                    # 前のデータと最も近い選手との距離を格納
                    min_difference_x = sequence_np[i, j, k] - sequence_np[i, j - 1, k]
                    min_difference_y = sequence_np[i, j, k + 1] - sequence_np[i, j - 1, k + 1]
                    min_difference_l = k
                    for l in range(int(number_of_player / 2)):
                        if (sequence_np[i, j, 4 * l + 3] == -1.0).any().any() or (sequence_np[i, j, 4 * l + 4] == -1.0).any().any():
                            continue
                        if (sequence_np[i, j, 4 * l + 3] - sequence_np[i, j - 1, k] < min_difference_x).any().any() and (sequence_np[i, j, 4 * l + 4] - sequence_np[i, j - 1, k + 1] < min_difference_y).any().any():
                            min_difference_x = sequence_np[i, j, 4 * l + 3] - sequence_np[i, j - 1, k]
                            min_difference_y = sequence_np[i, j, 4 * l + 3] - sequence_np[i, j - 1, k + 1]
                            min_difference_l = l
                    sequence_np[i, j, k] = sequence_np[i, j, 4 * min_difference_l + 3]
                    sequence_np[i, j, k + 1] = sequence_np[i, j, 4 * min_difference_l + 3]
                    print(sequence_np[i, j, k], sequence_np[i, j - 1, k], sequence_np[i, j, k + 1], sequence_np[i, j - 1, k + 1])


# count_player
def count_player(dir, file_length):
    count_teammate = np.zeros(12)
    count_opponent = np.zeros(12)

    for i in range(file_length):

        df = pd.read_csv(dir + "\\" + str(i + 1).zfill(6) + ".csv")

        for j in range(len(df)):
            if df.at[j,'teammate_1_x'] == -1.0:
                count_teammate[0] += 1
            elif df.at[j,'teammate_2_x'] == -1.0:
                count_teammate[1] += 1 
            elif df.at[j,'teammate_3_x'] == -1.0:
                count_teammate[2] += 1 
            elif df.at[j,'teammate_4_x'] == -1.0:
                count_teammate[3] += 1
            elif df.at[j,'teammate_5_x'] == -1.0:
                count_teammate[4] += 1
            elif df.at[j,'teammate_6_x'] == -1.0:
                count_teammate[5] += 1
            elif df.at[j,'teammate_7_x'] == -1.0:
                count_teammate[6] += 1
            elif df.at[j,'teammate_8_x'] == -1.0:
                count_teammate[7] += 1
            elif df.at[j,'teammate_9_x'] == -1.0:
                count_teammate[8] += 1
            elif df.at[j,'teammate_10_x'] == -1.0:
                count_teammate[9] += 1
            elif df.at[j,'teammate_11_x'] == -1.0:
                count_teammate[10] += 1
            else:
                count_teammate[11] += 1
            
            if df.at[j,'opponent_player_1_x'] == -1.0:
                count_opponent[0] += 1
            elif df.at[j,'opponent_player_2_x'] == -1.0:
                count_opponent[1] += 1 
            elif df.at[j,'opponent_player_3_x'] == -1.0:
                count_opponent[2] += 1 
            elif df.at[j,'opponent_player_4_x'] == -1.0:
                count_opponent[3] += 1
            elif df.at[j,'opponent_player_5_x'] == -1.0:
                count_opponent[4] += 1
            elif df.at[j,'opponent_player_6_x'] == -1.0:
                count_opponent[5] += 1
            elif df.at[j,'opponent_player_7_x'] == -1.0:
                count_opponent[6] += 1
            elif df.at[j,'opponent_player_8_x'] == -1.0:
                count_opponent[7] += 1
            elif df.at[j,'opponent_player_9_x'] == -1.0:
                count_opponent[8] += 1
            elif df.at[j,'opponent_player_10_x'] == -1.0:
                count_opponent[9] += 1
            elif df.at[j,'opponent_player_10_x'] == -1.0:
                count_opponent[10] += 1
            else:
                count_opponent[11] += 1
    
    print(count_teammate, count_opponent)

    csv1_path = r"c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\360_data\\count_teammate.csv"
    csv2_path = r"c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\360_data\\count_opponent.csv"

    np.savetxt(csv1_path, count_teammate, fmt='%s', delimiter=',')
    np.savetxt(csv2_path, count_opponent, fmt='%s', delimiter=',')


def sort_player(df):
    for j in range(len(df)):
        for k in range(12):
            teammate_x_list = [df.iat[j, 4 * k + 3]]
            
        teammate_x_list.sort(reverse=True)

        for k in range(12):
            df.iat[j, 4 * k + 3] = teammate_x_list[k]



if __name__ == "__main__":
    main()