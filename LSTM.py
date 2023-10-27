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

# device = 'cuda:0' if torch.cuda.is_available else 'cpu'


def main():

    '''labels = "shortcounter"

    competition_name = "LaLiga"

    # ラベル変更かつ全部同じファイルに
    label(labels,competition_name)'''


    # data_length
    data_length = 5


    '''dir = "c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\all"
    
    # ファイルの長さ
    file_length = count_file(dir)

    # sequenceの長さ
    sequence_length(dir, file_length)

    # sequenceのnumpyを作成
    sequence_np = np.zeros([file_length,data_length,3])

    # labelのnumpyを作成
    label_np = np.zeros([file_length])

    # ランダムに取り出す
    # random_k = rand_ints_nodup(1, file_length, file_length)

    for i in range(file_length):

        df = pd.read_csv(dir + "\\" + str(i + 1).zfill(6) + ".csv")


        # ラベルに応じてtrain_tにラベル付与
        if df.at[0,"label"] == 1:
            label_np[i] = 1
        elif df.at[0,"label"] == 2:
            label_np[i] = 2


        if len(df) <= data_length:
            for j in range(len(df)):
                sequence_np[i,j,0] = df.at[j,'start_x']
                sequence_np[i,j,1] = df.at[j,'start_y']
                sequence_np[i,j,2] = df.at[j,'time_seconds'] / 100
        else:
            for j in range(data_length):
                sequence_np[i,j,0] = df.at[j,'start_x']
                sequence_np[i,j,1] = df.at[j,'start_y']
                sequence_np[i,j,2] = df.at[j,'time_seconds'] / 100


        if i % 1000 == 0:
            print(i)
    
    # 転置
    label_np = label_np.T

    # numpy保存
    np.save('c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\sequence_label_np\\sequence_np_' + str(data_length), sequence_np)
    np.save('c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\sequence_label_np\\label_np_' + str(data_length), label_np)'''




    # numpy load
    sequence_np = np.load('c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\sequence_label_np\\sequence_np_' + str(data_length) + '.npy')
    label_np = np.load('c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\sequence_label_np\\label_np_' + str(data_length) + '.npy')

    
    LSTM(sequence_np,label_np)




def LSTM(train_x,train_t):
    # 入力


    # torch.tensorでtensor型に
    train_x = torch.from_numpy(train_x.astype(np.float32)).clone()
    train_t = torch.from_numpy(train_t.astype(np.int32)).clone()


    dataset = torch.utils.data.TensorDataset(train_x, train_t)


    # trainとvalidationをsplit
    n_samples = len(dataset)
    train_size = int(len(dataset) * 0.6) # train_size is 4800
    val_size = int(len(dataset) * 0.2) # val_size is 1600
    test_size = n_samples - train_size - val_size# val_size is 1600
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(42))



    batch_size = 512
    
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = False, num_workers = 0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False, num_workers = 0)
    

    model = LSTMClassification(input_dim=3, 
                            hidden_dim=6, 
                            target_size=3)
    
    
    epoch = 1000

    train(model, epoch, trainloader)

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

    model.load_state_dict(torch.load(PATH))

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

            inputs,labels = inputs.to(device), labels.to(device)

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
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs,labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
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
        if i >= 1000:
            break
        df = pd.read_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\" + str(labels) + "\\" + str(competition_name) + "\\" + str(i + 1).zfill(6) + ".csv")
        if labels == "no_counter":
            df["label"] = 0
        elif labels == "longcounter":
            df["label"] = 1
        elif labels == "shortcounter":
            df["label"] = 2
        else:
            break

        # ファイル移動
        # 前のラベルが何個入ったか
        s = i + 1000
        df.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\long_short_same\\" + str(s + 1).zfill(6) + ".csv")
        if i % 1000 == 0:
            print(i)



# ディレクトリ内のファイル数調査
def count_file(a):# labels,competition_name
    # dir = "C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\" + str(labels) + "\\" + str(competition_name) 
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
        '''elif (len(df) >= 1) and (len(df) <= 5):
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
            f += 1'''
        
        if i % 1000 == 0:
            print(i)

    print(max_len)

    '''sequence_length_list.append(a)
    sequence_length_list.append(b)
    sequence_length_list.append(c)
    sequence_length_list.append(d)
    sequence_length_list.append(e)
    sequence_length_list.append(f)
    
    csv_path = r"c:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\only_ball\\sequence_length.csv"

    sequence_length_np = np.array(sequence_length_list)

    np.savetxt(csv_path, sequence_length_np, fmt='%s', delimiter=',')'''



# csv to list
def read_csv(file_path):
    csv_file = open(file_path)              # csvファイルを開く
    csv_reader = csv.reader(csv_file)       # 開いたcsvファイルからreaderオブジェクトを生成

    date_list = []                          # 抽出するデータを格納する空のリストを作る。

    for row in csv_reader:                  # readerオブジェクトをループしてデータ抽出。
        if csv_reader.line_num == 1:        # ヘッダー行はスキップする。
            continue                        # Trueになる1行目はなにもしない。
        date_list.append(row[0])            # row行目の日時データをdate_listに格納する。    # row行目の気温データをdate_listに格納する。

    csv_file.close()                        # csvファイルを閉じる。
    return date_list     # 作成した２つのリスト(date_list, temperature_listを返す。)


# 重複なし
def rand_ints_nodup(a, b, k):
  ns = []
  while len(ns) < k:
    n = random.randint(a, b)
    if not n in ns:
      ns.append(n)
  return ns



# 360データをボールのみデータに
def three_sixty_to_only_ball(df):
    df = df.drop('360_data',axis = 1)
    return df

# 360データの整形
def three_sixty_data(df):
    i = 0
    for i in range(len(df) - 1):
        if df.loc[i,['360_data']] == None:
            break
        else:
            sub_list = df.loc[i,['360_data']]
            j = 0
            for j in range(len(sub_list) - 1):
                if sub_list[i]['actor'] == True:
                    continue                


if __name__ == "__main__":
    main()
