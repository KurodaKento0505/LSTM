'''ライブラリの準備'''
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.utils.rnn as rnn
import torchvision
from torchvision.transforms import functional as TF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import glob
import cv2
import optuna
import csv
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt


'''GPUチェック'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def CNN(sequence_np, label_np, number_of_player, number_of_tactical_action, test, make_graph, val):
    # 計算環境が、CUDA(GPU)か、CPUか
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ' + device)

    # 学習・テスト結果の保存用辞書
    history_train = {
        'train_loss': [],   # 損失関数の値
        'train_acc': [],    # 正解率
    }

    history_test = {
        'test_loss': [],    # 損失関数の値
        'test_acc': [],     # 正解率
    }

    # ネットワークを構築（ : torch.nn.Module は型アノテーション）
    # 変数netに構築するMyNet()は「ネットワークの実装」で定義します
    # net : torch.nn.Module = MyNet()

    cnn_output_dim = 20 # CNNの出力次元数
    hidden_dim = 20  # LSTMの隠れ層の次元数
    num_classes = number_of_tactical_action  # クラス数（ラベルの種類数）
    net = CNNLSTM(cnn_output_dim, hidden_dim, num_classes)
    net = net.to(device) # GPUあるいはCPUに合わせて再構成

    # データローダー・データ数を取得
    #（load_dataは「学習データ・テストデータのロードの実装（データローダー）」の章で定義します）
    train_loaders, train_data_size = load_data(sequence_np, label_np)
    print(train_loaders, train_data_size)
    # test_loaders, test_data_size = load_data(sequence_np, label_np)
    
    # オプティマイザを設定
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    
    # エポック数（学習回数）
    epoch = 1000

    # 学習・テストを実行
    for e in range(epoch):
        # 以下2つの関数は「学習の実装」「テストの実装」の章で定義します
        cnn_train(net, device, train_loaders, train_data_size, optimizer, e, history_train)
        # cnn_test(net, device, test_loaders, test_data_size, e, epoch, history_test)

    # 学習済みパラメータを保存
    torch.save(net.state_dict(), 'params_cnn.pth')

    # 結果を出力（「結果出力の実装」の章で定義します）
    # output_graph(epoch, history_train, history_test)



# モデルの定義
class CNNLSTM(nn.Module):
    def __init__(self, cnn_output_dim, hidden_dim, num_classes):
        super(CNNLSTM, self).__init__()
        self.cnn_output_dim = cnn_output_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        input_dim=(28, 42)
        filter_num = 30 # フィルター数
        filter_size = 5 # フィルター（カーネル）サイズ

        # CNN部分
        # conv2d定義
        self.conv1 = torch.nn.Conv2d(in_channels=input_dim[0], out_channels=filter_num, kernel_size=filter_size)
    
        # ReLU定義
        self.relu = torch.nn.ReLU()

        # Pooling定義
        pooling_size = 2
        self.pool = torch.nn.MaxPool2d(pooling_size, stride=pooling_size)

        # Affine（全結合）レイヤ定義
        fc1_size = (input_dim[1] - filter_size + 1) // pooling_size # self.pool終了時の縦横サイズ = 12
        self.fc1 = torch.nn.Linear(filter_num * fc1_size * fc1_size, 100)   # ノード100個に出力
        self.fc2 = torch.nn.Linear(100, 10)                         # ノード10個に出力 
        
        # LSTM部分
        self.lstm = nn.LSTM(cnn_output_dim, hidden_dim, batch_first=True)
        
        # 分類器部分
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, height, width = x.size() # channels, 
        x = x.view(batch_size * seq_length, height, width) # channels, 
        # conv2d - ReLU - Poolingまで
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 全結合層に入れるため、バッチ内の各画像データをそれぞれ一列にする
        x = x.view(x.size()[0], -1) 
        
        # Affine - ReLU - Affine
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(x)
        # LSTMの最後の出力のみを使う場合
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output



class MyNet(torch.nn.Module):
    """
    CNNを用いた、オライリーP.229 7-5章と同じモデル。
    バッチの画像群に対し、
    入力(batch_size, 3, 28, 28) - conv2d - ReLU - Pooling - Affine - ReLU - Affine - Softmax
    を計算する。

    Returns:
        float: 10個の要素に対する確率を格納した配列。
    """
    
    def __init__(self, input_dim=(3, 28, 28)):
        """
        ネットワークで使用する関数の定義。
        """
        # モジュールの継承
        super(MyNet, self).__init__()
        
        filter_num = 30 # フィルター数
        filter_size = 5 # フィルター（カーネル）サイズ

        # conv2d定義
        self.conv1 = torch.nn.Conv2d(in_channels=input_dim[0], out_channels=filter_num, kernel_size=filter_size)
    
        # ReLU定義
        self.relu = torch.nn.ReLU()

        # Pooling定義
        pooling_size = 2
        self.pool = torch.nn.MaxPool2d(pooling_size, stride=pooling_size)

        # Affine（全結合）レイヤ定義
        fc1_size = (input_dim[1] - filter_size + 1) // pooling_size # self.pool終了時の縦横サイズ = 12
        self.fc1 = torch.nn.Linear(filter_num * fc1_size * fc1_size, 100)   # ノード100個に出力
        self.fc2 = torch.nn.Linear(100, 10)                         # ノード10個に出力
    
    def forward(self, x):
        """
        順伝播の定義。
        """
        # conv2d - ReLU - Poolingまで
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 全結合層に入れるため、バッチ内の各画像データをそれぞれ一列にする
        x = x.view(x.size()[0], -1) 
        
        # Affine - ReLU - Affine
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # ソフトマックスにより、各ノードの確率を結果として返す
        return f.softmax(x, dim=1) # dim=1はバッチ内の各画像データごとにsoftmaxをするため



# Main()で、画像データがあるディレクトリパスをdir_pathとして受け取る
def load_data(sequence_np, label_np):
    """画像データを読み込み、データセットを作成する。
    画像のファイル名は、末尾を「...ans〇.jpg」（〇に正解ラベルの数字を入れる）として用意すること。 

    Returns:
        DataLoader: 画像データ配列と正解ラベル配列がバッチごとにペアになったイテレータ
        data_size: データ数
    """

    '''# 画像ファイル名を全て取得
    img_paths = os.path.join(dir_path, "*.jpg")
    img_path_list = glob.glob(img_paths)'''

    # 画像データ・正解ラベル格納用配列
    data = sequence_np
    labels = label_np

    '''# 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像読み込み・(3, height, width)に転置・正規化
        img = TF.to_tensor(cv2.imread(img_path))

        # 画像をdataにセット
        data.append(img.detach().numpy()) # 配列にappendするため、一度ndarray型へ

        # 正解ラベルをlabelsにセット
        begin = img_path.find('ans') + len('ans')
        tail = img_path.find('.jpg')
        ans = int(img_path[begin:tail])
        labels.append(ans)'''

    # PyTorchで扱うため、tensor型にする
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    
    # 画像データ・正解ラベルのペアをデータにセットする
    dataset = torch.utils.data.TensorDataset(data, labels)
    
    # セットしたデータをバッチサイズごとの配列に入れる。
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # データ数を取得
    data_size = len(sequence_np)
    
    return loader, data_size



def cnn_train(net, device, loaders, data_size, optimizer, e, history):
    """CNNによる学習を実行する。
    net.parameters()に各conv, fcのウェイト・バイアスが格納される。
    """

    loss = None         # 損失関数の結果
    loss_sum = 0        # 損失関数の値（エポック合計）
    train_correct_counter = 0   # 正解数カウント

    # 学習開始（再開）
    net.train(True) # 引数は省略可能

    for i, (data, labels) in enumerate(loaders):
        # GPUあるいはCPU用に再構成
        data = data.to(device)      # バッチサイズの画像データtensor
        labels = labels.to(device)  # バッチサイズの正解ラベルtensor

        # 学習
        optimizer.zero_grad()   # 前回までの誤差逆伝播の勾配をリセット
        print(data, labels)
        output = net(data)      # 推論を実施（順伝播による出力）

        loss = f.nll_loss(output, labels)   # 交差エントロピーによる損失計算（バッチ平均値）
        loss_sum += loss.item() * data.size()[0] # バッチ合計値に直して加算

        loss.backward()         # 誤差逆伝播
        optimizer.step()        # パラメータ更新

        train_pred = output.argmax(dim=1, keepdim=True) # 0 ~ 9のインデックス番号がそのまま推論結果
        train_correct_counter += train_pred.eq(labels.view_as(train_pred)).sum().item() # 推論と答えを比較し、正解数を加算

        # 進捗を出力（8バッチ分ごと）
        if i % 8 == 0:
            print('Training log: epoch_{} ({} / {}). Loss: {}'.format(e+1, (i+1)*loaders.batch_size, data_size, loss.item()))

    # エポック全体の平均の損失関数、正解率を格納
    ave_loss = loss_sum / data_size
    ave_accuracy = train_correct_counter / data_size
    history['train_loss'].append(ave_loss)
    history['train_acc'].append(ave_accuracy)
    print(f"Train Loss: {ave_loss} , Accuracy: {ave_accuracy}")

    return



def cnn_test(net, device, loaders, data_size, e, epoch, history):
    """
    学習したパラメータでテストを実施する。
    """
    # 学習のストップ
    net.eval() # または　net.train(False)でもいい

    loss_sum = 0                # 損失関数の値（数値のみ）
    test_correct_counter = 0    # 正解数カウント
    data_num = 0                # 最終エポックでの出力画像用ナンバー

    with torch.no_grad():
        for data, labels in loaders:
            # GPUあるいはCPU用に再構成
            data = data.to(device)      # バッチサイズの画像データtensor
            labels = labels.to(device)  # バッチサイズの正解ラベルtensor

            output = net(data)  # 推論を実施（順伝播による出力）
            loss_sum += f.nll_loss(output, labels, reduction='sum').item() # 損失計算　バッチ内の合計値を加算

            test_pred = output.argmax(dim=1, keepdim=True) # 0 ~ 9のインデックス番号がそのまま推論結果
            test_correct_counter += test_pred.eq(labels.view_as(test_pred)).sum().item() # 推論と答えを比較し、正解数を加算

            # 最終エポックのみNG画像を出力
            if e == epoch - 1:
                last_epoch_NG_output(data, test_pred, labels, data_num)
                data_num += loaders.batch_size
    
    # テスト全体の平均の損失関数、正解率を格納
    ave_loss = loss_sum / data_size
    ave_accuracy = test_correct_counter / data_size
    history['test_loss'].append(ave_loss)
    history['test_acc'].append(ave_accuracy)
    print(f'Test Loss: {ave_loss} , Accuracy: {ave_accuracy}\n')

    return

def last_epoch_NG_output(data, test_pred, target, counter):
    """
    不正解した画像を出力する。
    ファイル名：「データ番号-pre-推論結果-ans-正解ラベル.jpg」
    """
    # フォルダがなければ作る
    dir_path = "./NG_photo_CNN"
    os.makedirs(dir_path, exist_ok=True)

    for i, img in enumerate(data):
        pred_num = test_pred[i].item()  # 推論結果
        ans = target[i].item()          # 正解ラベル

        # 推論結果と正解ラベルを比較して不正解なら画像保存
        if pred_num != ans:
            # ファイル名設定
            data_num = str(counter+i).zfill(5)
            img_name = f"{data_num}-pre-{pred_num}-ans-{ans}.jpg"
            fname = os.path.join(dir_path, img_name)
            
            # 画像保存
            torchvision.utils.save_image(img, fname)

    return



def output_graph(epoch, history_train, history_test):
    os.makedirs("./CNNLearningResult", exist_ok=True)

    # 各エポックの損失関数グラフ
    plt.figure()
    plt.plot(range(1, epoch+1), history_train['train_loss'], label='train_loss', marker='.')
    plt.plot(range(1, epoch+1), history_test['test_loss'], label='test_loss', marker='.')
    plt.xlabel('epoch')
    plt.legend() # 凡例
    plt.savefig('./CNNLearningResult/loss_cnn.png')

    # 各エポックの正解率グラフ
    plt.figure()
    plt.plot(range(1, epoch+1), history_train['train_acc'], label='train_acc', marker='.')
    plt.plot(range(1, epoch+1), history_test['test_acc'], label='test_acc', marker='.')
    plt.xlabel('epoch')
    plt.legend() # 凡例
    plt.savefig('./CNNLearningResult/acc_cnn.png')

    return
