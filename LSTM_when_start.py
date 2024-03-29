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

'''GPUチェック'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import csv
import pandas as pd
import os
import random
import numpy as np
from func_heatmap import make_heatmap

'''from func_make_graph import make_graph
from func_make_vector import make_vector
from LSTM_percent import count_player'''


# what kind of data
data = 'heatmap' # only_x_3_players only_x_6_players only_x_10_players only_x_no_player both_xy_no_player 

competition = "FIFA_World_Cup_2022" # FIFA_World_Cup_2022, UEFA_Euro_2020, UEFA_Women's_Euro_2022, Women's_World_Cup_2023, _convert
print(competition)


def main():


    '''labels = "shortcounter" # longcounter shortcounter opposition_half_possession own_half_possession others

    competition_name = "Women's_World_Cup_2023"  # FIFA_World_Cup_2022, UEFA_Euro_2020, UEFA_Women's_Euro_2022, Women's_World_Cup_2023, La_Liga

    # ラベル変更かつ全部同じファイルに
    label(labels,competition_name)'''

    # data_length
    data_length = 11

    # number of player
    number_of_player = 10

    # 戦術的行動の数
    number_of_tactical_action = 4

    # test か否か
    test = False

    # val か否か
    val = False

    # make_graph か否か
    make_graph = True


    # train
    if test != True:
        dir = "C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence_choice_42000"
    # test
    else:
        dir = "C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\test_sequence\\" + competition # _convert

    # make graph
    # make_graph(dir_graph)
    
    # ファイルの長さ
    file_length = count_file(dir)

    # 格子のサイズ
    grid_size_x = 121  # x座標の最大値 + 1
    grid_size_y = 81   # y座標の最大値 + 1

    # only defense と only no counter を除く
    # sequence_choice(dir, file_length)

    # sequenceの長さ
    # sequence_length(dir, file_length)

    # sequenceのnumpyを作成
    # （ボール + 選手22人）x 2 + 時間 + ラベル = 44
    # sequence_np = np.full([file_length, data_length, number_of_player * 1 + 2], 0.0) # ベクトルの時は、data_length - 1  # only_x の時は * 1 + 2
    # 画像の場合
    # RGB
    # sequence_np = np.zeros([file_length, data_length, grid_size_x, grid_size_y, 3])
    # sequence_np = np.empty((data_length, grid_size_x, grid_size_y, 3), float)
    # gray
    sequence_np = np.empty((file_length, data_length, grid_size_y, grid_size_x), float)
    # labelのnumpyを作成
    label_np = np.full([file_length, number_of_tactical_action], 0.0) # 離散値の場合：１，連続値の場合：0.0


    # count_player
    # count_player(dir, file_length)
    

    for i in range(file_length):
        
        df = pd.read_csv(dir + "\\" + str(i + 1).zfill(6) + ".csv")

        # make_heatmap
        image_sequence_np = make_heatmap(df, data_length, grid_size_x, grid_size_y)
        sequence_np[i] = image_sequence_np
        # sequence_np = np.append(image_sequence_np, sequence_np, axis=0)
        
        # make_vector
        # make_vector(df)

        # until counter をlabel_npに格納
        # put_in_label(df, label_np, i)

        # put in segmentstion data from df to sequence_np
        # put_in_seg_data(df, sequence_np, i, number_of_player)
        
        # put in attack_sequence data from df to sequence_np
        # put_in_360_data(df, sequence_np, data_length, i)


        # 360 data 整形
        # indentify_player(sequence_np, number_of_player, data_length, i)


        if i % 1000 == 0:
            print(i)
    
    # 転置
    # label_np = label_np.T

    # print(sequence_np)
    # print(label_np)


    # numpy保存
    # train
    if test != True:
        np.save('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\sequence_np\\' + data + '_sequence_np', sequence_np) # test_1_ 0_to_2_ 5000_
        # np.save('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\label_np\\' + data + '_label_np', label_np)
    else:
        np.save("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\test_data\\" + competition + "\\" + data + "_sequence_np", sequence_np) # test_1_ 0_to_2_ 5000_
        np.save("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\test_data\\" + competition + "\\" + data + "_label_np", label_np)


    '''# numpy load
    # train
    if test != True:
        sequence_np = np.load('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\sequence_np\\' + data + '_sequence_np.npy') # \\test test_ vector_ include_possession_
        label_np = np.load('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\label_np\\' + data + '_label_np.npy') # discrete_
    
    # test
    else:
        if make_graph:
            # make_graph
            sequence_np = np.load("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\test_data\\" + competition + "\\" + data + '_sequence_np.npy') # \\test test_ vector_ include_possession_
            label_np = np.load("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\test_data\\" + competition + "\\" + data + "_label_np.npy")

        else:
            # random_test_data
            sequence_np = np.load('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\sequence_np\\' + data + '_sequence_np.npy') # \\test test_ vector_ include_possession_
            label_np = np.load('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\label_np\\' + data + '_label_np.npy') # discrete_
'''


    '''# ラベルごとに分割
    # Timeありかなしかで 2 or 3
    sequence_0_np = np.zeros([76098, data_length, number_of_player * 2 + 3], dtype = np.float16)
    sequence_1_np = np.zeros([848, data_length, number_of_player * 2 + 3], dtype = np.float16)
    sequence_2_np = np.zeros([1013, data_length, number_of_player * 2 + 3], dtype = np.float16)

    label_0_np = np.zeros([76098])
    label_1_np = np.zeros([848])
    label_2_np = np.zeros([1013])'''

    '''only_defense = 0 # 守備の数
    longcounter = 0 # ロングカウンターの数
    shortcounter = 0 # ショートカウンターの数
    only_no_counter = 0
    defense_or_no_counter = 0

    for i in range(len(label_np)):
        if label_np[i, 0] == 1:
            only_defense += 1
        elif label_np[i, 3] == 1:
            only_no_counter += 1
        elif label_np[i, 1] == 0 and label_np[i, 2] == 0:
            defense_or_no_counter += 1
        elif label_np[i, 1] != 0:
            longcounter += 1
        elif label_np[i, 2] != 0:
            shortcounter += 1

    print(only_defense, only_no_counter, defense_or_no_counter, longcounter, shortcounter)
    print(only_defense + only_no_counter + defense_or_no_counter + longcounter + shortcounter)'''

    #LSTM(sequence_np, label_np, number_of_player, number_of_tactical_action, test, make_graph, val)





def LSTM(train_x, train_t, number_of_player, number_of_tactical_action, test, make_graph, val): 

    # torch.tensorでtensor型に
    train_x = torch.from_numpy(train_x.astype(np.float32)).clone()
    train_t = torch.from_numpy(train_t.astype(np.float32)).clone()


    batch_size = 512
    hidden_dim = 20
    epoch = 1000
    lr = 0.1


    if test != True:
        dataset = torch.utils.data.TensorDataset(train_x, train_t)

        train_size = int(len(dataset) * 0.6) # train_size is 3000
        val_size = int(len(dataset) * 0.2) # val_size is 1000
        test_size = int(len(dataset) * 0.2)# val_size is 1000
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
    model = LSTMClassification(input_dim = number_of_player * 1 + 2, 
                            hidden_dim = hidden_dim, 
                            target_size = number_of_tactical_action)


    PATH = './cifar_net.pth'

    if test != True:
        train(model, epoch, trainloader, valloader, lr)
        torch.save(model.state_dict(), PATH)
        # evaluate(model, testloader, test)

    else:
        model.load_state_dict(torch.load(PATH))

        if make_graph:
            evaluate(model, graph_testloader, test, number_of_tactical_action, make_graph, val)
        else:
            if val:
                evaluate(model, valloader, test, number_of_tactical_action, make_graph, val)
            else:
                evaluate(model, testloader, test, number_of_tactical_action, make_graph, val)


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
    loss_function = nn.HuberLoss() # SmoothL1
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



def evaluate(model, loader, test, number_of_tactical_action, make_graph, val):
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

        percent_calculate(outputs_list, labels_list, len(loader), test, number_of_tactical_action, make_graph, val)
    # accuracy = correct / total
    # return accuracy



# modelの評価
def percent_calculate(outputs, labels, length, test, number_of_tactical_action, make_graph, val):

    length = len(outputs)
    outputs_np = np.array(outputs)
    labels_np = np.array(labels)

    print(labels_np)
    print(outputs_np)


    if test:

        # optunaによる閾値決定が終わったらメモした値を代入
        # threhold を格納
        threshold = np.zeros(number_of_tactical_action) # 0 人

        threshold[0] = 0.8111589377624764
        threshold[1] = 0.6870384882520266
        threshold[2] = 0.5289033916825636
        threshold[3] = 0.023588469646860406

        if make_graph:
            # make_graph
            dir = "C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\test_sequence\\" + competition

            df_make_graph = pd.DataFrame()

            file_length = count_file(dir)

            for i in range(file_length):

                df = pd.read_csv(dir + "\\" + str(i + 1).zfill(6) + ".csv")

                df['output_until_longcounter'] = outputs_np[i, 0]
                df['output_until_shortcounter'] = outputs_np[i, 1]
                df['output_until_opposition_half_possession'] = outputs_np[i, 2]
                df['output_until_own_half_possession'] = outputs_np[i, 3]
                # df['output_until_others'] = outputs_np[i, 4]

                df_make_graph.loc[i, ['start_x', 'start_y', 'time_seconds', 'label', 'output_until_longcounter', 'output_until_shortcounter', 'output_until_opposition_half_possession', 'output_until_own_half_possession']] = df.loc[5, ['start_x', 'start_y', 'time_seconds', 'label', 'output_until_longcounter', 'output_until_shortcounter', 'output_until_opposition_half_possession', 'output_until_own_half_possession']]
                df_make_graph.loc[i, ['until_longcounter', 'until_shortcounter', 'until_opposition_half_possession', 'until_own_half_possession']] = df.loc[0, ['until_longcounter', 'until_shortcounter', 'until_opposition_half_possession', 'until_own_half_possession']]

            df_make_graph.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\test_data\\" + competition + "\\" + data + "_outputs.csv")


            # 前半と後半を区別
            for i in range(len(df_make_graph)):

                if i == 0:
                    continue

                if (df_make_graph.loc[i, ['time_seconds']] - df_make_graph.loc[i - 1, ['time_seconds']] < 0).any().any():

                    # 前半
                    df_make_graph_1st_half = df_make_graph[ : i - 1].copy()
                    df_make_graph_1st_half.reset_index(drop=True, inplace=True)

                    # 後半
                    df_make_graph_2nd_half = df_make_graph[i : ].copy()
                    df_make_graph_2nd_half.reset_index(drop=True, inplace=True)

                    break


            # 各戦術的行動の名前
            tactical_action_name_list = ['longcounter', 'shortcounter', 'opposition_half_possession', 'own_half_possession']
            
            # 各戦術的行動の正解データの数
            number_of_tactical_action_np = np.zeros(4)

            # 各戦術的行動の正解データの中で認識できた数
            # threshold_1
            recognized_1_tactical_action_np = np.zeros(4)
            # threshold_2
            recognized_2_tactical_action_np = np.zeros(4)


            # 前半の認識精度を計算
            for i in range(4, len(df_make_graph_1st_half) - 4):

                for j in range(4):

                    # 正解データにおいて, 時間距離0になる時間を探す
                    # longcounter
                    if (df_make_graph_1st_half.loc[i, ['until_' + tactical_action_name_list[j]]] == 0).any().any():

                        number_of_tactical_action_np[j] += 1

                        # threshold_1
                        for k in range(i - 4, i + 4):

                            if (df_make_graph_1st_half.loc[k, ['output_until_' + tactical_action_name_list[j]]] <= threshold[j]).any().any():
                                recognized_1_tactical_action_np[j] += 1
                                break

            # 後半の認識精度を計算
            for i in range(4, len(df_make_graph_2nd_half) - 4):

                for j in range(4):

                    # 正解データにおいて, 時間距離0になる時間を探す
                    # longcounter
                    if (df_make_graph_2nd_half.loc[i, ['until_' + tactical_action_name_list[j]]] == 0).any().any():

                        number_of_tactical_action_np[j] += 1

                        # threshold_1
                        for k in range(i - 4, i + 4):

                            if (df_make_graph_2nd_half.loc[k, ['output_until_' + tactical_action_name_list[j]]] <= threshold[j]).any().any():
                                recognized_1_tactical_action_np[j] += 1
                                break

            print(number_of_tactical_action_np)
            print(recognized_1_tactical_action_np)

        else:
            if val:
                # optunaによる閾値決定(val)
                # ラベルを０か１に
                longcounter_label_0_or_1_np = labels_np[:,0].astype(np.float32)
                shortcounter_label_0_or_1_np = labels_np[:,1].astype(np.float32)
                opposition_half_possession_label_0_or_1_np = labels_np[:,2].astype(np.float32)
                own_half_possession_label_0_or_1_np = labels_np[:,3].astype(np.float32)

                longcounter_output_0_or_1_np = outputs_np[:,0].astype(np.float32)
                shortcounter_output_0_or_1_np = outputs_np[:,1].astype(np.float32)
                opposition_half_possession_output_0_or_1_np = outputs_np[:,2].astype(np.float32)
                own_half_possession_output_0_or_1_np = outputs_np[:,3].astype(np.float32)

                test_length = length

                for i in range(length):
                        if longcounter_label_0_or_1_np[i] != 1:
                            longcounter_label_0_or_1_np[i] = 0

                        if shortcounter_label_0_or_1_np[i] != 1:
                            shortcounter_label_0_or_1_np[i] = 0

                        if opposition_half_possession_label_0_or_1_np[i] != 1:
                            opposition_half_possession_label_0_or_1_np[i] = 0

                        if own_half_possession_label_0_or_1_np[i] != 1:
                            own_half_possession_label_0_or_1_np[i] = 0

                # threhold を格納
                threshold = np.zeros(number_of_tactical_action)
                
                study= optuna.create_study(direction="maximize") 
                study.optimize(objective_longcounter_variable_degree(longcounter_label_0_or_1_np, longcounter_output_0_or_1_np), n_trials = 100)
                print(study.best_params)
                threshold[0] = study.best_params['threshold']

                study = optuna.create_study(direction="maximize") 
                study.optimize(objective_shortcounter_variable_degree(shortcounter_label_0_or_1_np, shortcounter_output_0_or_1_np), n_trials = 100)
                print(study.best_params)
                threshold[1] = study.best_params['threshold']

                study = optuna.create_study(direction="maximize") 
                study.optimize(objective_opposition_half_possession_variable_degree(opposition_half_possession_label_0_or_1_np, opposition_half_possession_output_0_or_1_np), n_trials = 100)
                print(study.best_params)
                threshold[2] = study.best_params['threshold']

                study = optuna.create_study(direction="maximize") 
                study.optimize(objective_own_half_possession_variable_degree(own_half_possession_label_0_or_1_np, own_half_possession_output_0_or_1_np), n_trials = 100)
                print(study.best_params)
                threshold[3] = study.best_params['threshold']

                for i in range(number_of_tactical_action):
                    print(threshold[i])


            else:
                # test_dataによる精度評価
                # binarization
                # label
                labels_binarization_np = np.full([length, number_of_tactical_action], 1)
                # 全ての戦術的行動のMAE出すための一次元配列
                all_labels_binarization_np = np.full([length * number_of_tactical_action], 1)
                for i in range(length):
                    for j in range(number_of_tactical_action):
                        if labels_np[i, j] != 1:
                            labels_binarization_np[i, j] = 0
                            all_labels_binarization_np[i * number_of_tactical_action + j] = 0
                        else:
                            labels_binarization_np[i, j] = 1
                            all_labels_binarization_np[i * number_of_tactical_action + j] = 1
                        if labels_np[i, j] != 1:
                            labels_binarization_np[i, j] = 0
                            all_labels_binarization_np[i * number_of_tactical_action + j] = 0

                # output
                outputs_binarization_np = np.full([length, number_of_tactical_action], 1)
                # 全ての戦術的行動のMAE出すための一次元配列
                all_outputs_binarization_np = np.full([length * number_of_tactical_action], 1)
                for i in range(length):
                    for j in range(number_of_tactical_action):
                        if outputs_np[i, j] <= threshold[j]:
                            outputs_binarization_np[i, j] = 0
                            all_outputs_binarization_np[i * number_of_tactical_action + j] = 0
                        else:
                            outputs_binarization_np[i, j] = 1
                            all_outputs_binarization_np[i * number_of_tactical_action + j] = 0
                        if outputs_np[i, j] <= threshold[j]:
                            outputs_binarization_np[i, j] = 0
                            all_outputs_binarization_np[i * number_of_tactical_action + j] = 0

                # 全ての戦術的行動のacc出すための一次元配列
                all_labels_np = np.full([length * number_of_tactical_action], 1)
                all_outputs_np = np.full([length * number_of_tactical_action], 1)
                for i in range(length):
                    for j in range(number_of_tactical_action):
                        all_labels_np[i * number_of_tactical_action + j] = labels_np[i, j]
                        all_outputs_np[i * number_of_tactical_action + j] = outputs_np[i, j]


                for i in range(number_of_tactical_action):
                    print(mean_absolute_error(labels_np[:, i], outputs_np[:, i]))
                print('\n')
                
                for i in range(number_of_tactical_action):
                    print(mean_squared_error(labels_np[:, i], outputs_np[:, i]))
                print('\n')
                
                for i in range(number_of_tactical_action):
                    print(f1_score(labels_binarization_np[:, i], outputs_binarization_np[:, i], pos_label = 0))
                print('\n')

                for i in range(number_of_tactical_action):
                    print(recall_score(labels_binarization_np[:, i], outputs_binarization_np[:, i], pos_label = 0))
                print('\n')

                for i in range(number_of_tactical_action):
                    print(accuracy_score(labels_binarization_np[:, i], outputs_binarization_np[:, i]))




# データにラベルをつける ＆ 同じファイルに番号変えて移動
def label(labels,competition_name):

    # ラベル付与
    i = 0
    for i in range(count_file(labels,competition_name)):
        if i >= 10000:
            break
        df = pd.read_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + str(competition_name) + "\\" + str(labels) + "\\" + str(i + 1).zfill(6) + ".csv")

        # ファイル移動
        # 前のラベルが何個入ったか
        s = i + 39372
        df.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence_choice_42000\\" + str(s + 1).zfill(6) + ".csv")
        
        print(str(s + 1).zfill(6))



# ディレクトリ内のファイル数調査
def count_file(a):# labels,competition_name  or  a  or  competition_name
    # dir = "C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + str(competition_name) + "\\" + str(labels)
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



def put_in_seg_data(df, sequence_np, i, number_of_player):

    for j in range(len(df)): # vectorの時は、len(df) - 1


        # 味方の前線 number_of_player/2 人を選ぶ

        # 宣言
        teammate_np = np.full(11, -1.0)
        forward_teammate_np = np.full(int(number_of_player/2), -1.0)

        # 人数確認
        # チームメートの人数11人の時, キーパーがボール保持者だから1人目除いて11人目含む, でも前線から選ぶからキーパーの心配はいらない, 相手の人数11人になることはない.
        if df.at[j, 'teammate_count'] == 0:
            pass

        else:
            # teamamte_np に 11 人のx位置を格納
            for k in range(11):
                teammate_np[k] = df.at[j,'teammate_' + str(k + 1) + '_x']
            
            # 降順に並べ替える
            teammate_np = np.sort(teammate_np)[::-1]

            # forward_teammate_np にnumber_of_player/2人格納
            for k in range(int(number_of_player/2)):
                forward_teammate_np[k] = teammate_np[k]

        
        # 相手のディフェンスライン number_of_player/2 人を選ぶ

        # 宣言
        # keeper は含まない
        opponent_np = np.full(10, -1.0)
        defense_opponent_np = np.full(int(number_of_player/2), -1.0)

        # 人数確認
        if df.at[j, 'opponent_count'] == 0:
            pass

        else:
            # opponent_np に 10 人のx位置を格納
            for k in range(10):
                opponent_np[k] = df.at[j,'opponent_' + str(k + 1) + '_x']
            
            # 降順に並べ替える
            opponent_np = np.sort(opponent_np)[::-1]

            # defense_opponent_np に number_of_player/2 人格納
            for k in range(int(number_of_player/2)):
                defense_opponent_np[k] = opponent_np[k]


        '''# basic (ball_x,y + time)
        sequence_np[i, j, 0] = df.at[j,'start_x']
        sequence_np[i, j, 1] = df.at[j,'start_y']
        sequence_np[i, j, 2] = df.at[j,'time_reset']'''

        # only_x + no_player 
        sequence_np[i, j, 0] = df.at[j,'start_x']
        sequence_np[i, j, 1] = df.at[j,'time_reset']


        '''# only_x + 3_teammate（前線3人）+ 3_opponent（守備3人）
        # only_x + 6_teammate（前線6人）+ 6_opponent（守備6人）
        # only_x + 10_teammate（前線10人）+ 10_opponent（守備10人）
        # teammate
        for l in range(2, 2 + int(number_of_player/2)):
            sequence_np[i, j, l] = forward_teammate_np[l - 2]

        # opponent
        for m in range(2 + int(number_of_player/2), 2 + number_of_player):
            sequence_np[i, j, m] = defense_opponent_np[m - 2 - int(number_of_player/2)]'''


        '''# x + y + tammate_mean + opponent_mean
        sequence_np[i, j, 0] = df.at[j,'start_x']
        sequence_np[i, j, 1] = df.at[j,'start_y']
        sequence_np[i, j, 2] = df.at[j,'time_reset'] # make graphのときのみ time_reset
        sequence_np[i, j, 3] = df.at[j,'teammate_mean_x']
        sequence_np[i, j, 4] = df.at[j,'teammate_mean_y']
        sequence_np[i, j, 5] = df.at[j,'opponent_mean_x']
        sequence_np[i, j, 6] = df.at[j,'opponent_mean_y']'''

        '''# only_x + tammate_mean + opponent_mean
        sequence_np[i, j, 0] = df.at[j,'start_x']
        sequence_np[i, j, 1] = df.at[j,'time_reset']
        sequence_np[i, j, 2] = df.at[j,'teammate_mean_x']
        sequence_np[i, j, 3] = df.at[j,'opponent_mean_x']'''

        '''# ベクトル（初期値なし）
        sequence_np[i, j, 0] = df.at[j,'pass_distance']
        sequence_np[i, j, 1] = df.at[j,'pass_radian']
        sequence_np[i, j, 2] = df.at[j,'pass_time']'''


def put_in_label(df, label_np, i): # 0 ~ 2 の時は + 1

    # 連続値による表現
    # 開始点探索
    label_np[i, 0] = df.at[0, 'until_longcounter']
    label_np[i, 1] = df.at[0, 'until_shortcounter']
    label_np[i, 2] = df.at[0, 'until_opposition_half_possession']
    label_np[i, 3] = df.at[0, 'until_own_half_possession']
    # label_np[i, 4] = df.at[0, 'until_others']
    
    '''# 離散値による表現
    if (df.at[5, 'label'] == 1).any().any():
        label_np[i, 0] = 0
    elif (df.at[5, 'label'] == 2).any().any():
        label_np[i, 1] = 0
    elif (df.at[5, 'label'] == 3).any().any():
        label_np[i, 2] = 0
    elif (df.at[5, 'label'] == 4).any().any():
        label_np[i, 3] = 0'''


# 閾値付きのrecall
def recall(y_test, y_prob, threshold):

    y_pred = np.zeros(len(y_prob))
    
    for i in range(len(y_prob)):
        if y_prob[i] >= threshold:
            y_pred[i] = 1

        if y_test[i] != 0 and y_test[i] != 1:
            y_test[i] = 0

    # score = recall_score(y_test.astype(int), y_pred.astype(int), pos_label = 0)
    # score = accuracy_score(y_test.astype(int), y_pred.astype(int))
    # score = precision_score(y_test.astype(int), y_pred.astype(int), pos_label = 0)
    score = f1_score(y_test.astype(int), y_pred.astype(int), pos_label = 0)

    return score


# 目的関数
# longcounter
def objective_longcounter_variable_degree(longcounter_label_0_or_1_np, longcounter_output_0_or_1_np):
    def objective_longcounter(trial): 
        threshold = trial.suggest_float('threshold', 0.0, 1.0) # 0~1.0で探索
        ret = recall(longcounter_label_0_or_1_np, longcounter_output_0_or_1_np, threshold)
        return ret
    return objective_longcounter

# shortcounter
def objective_shortcounter_variable_degree(shortcounter_label_0_or_1_np, shortcounter_output_0_or_1_np):
    def objective_shortcounter(trial): 
        threshold = trial.suggest_float('threshold', 0.0, 1.0) # 0~1.0で探索
        ret = recall(shortcounter_label_0_or_1_np, shortcounter_output_0_or_1_np, threshold)
        return ret
    return objective_shortcounter

# opposition_half_possession
def objective_opposition_half_possession_variable_degree(opposition_half_possession_label_0_or_1_np, opposition_half_possession_output_0_or_1_np):
    def objective_opposition_half_possession(trial): 
        threshold = trial.suggest_float('threshold', 0.0, 1.0) # 0~1.0で探索
        ret = recall(opposition_half_possession_label_0_or_1_np, opposition_half_possession_output_0_or_1_np, threshold)
        return ret
    return objective_opposition_half_possession

# own_half_possession
def objective_own_half_possession_variable_degree(own_half_possession_label_0_or_1_np, own_half_possession_output_0_or_1_np):
    def objective_own_half_possession(trial): 
        threshold = trial.suggest_float('threshold', 0.0, 1.0) # 0~1.0で探索
        ret = recall(own_half_possession_label_0_or_1_np, own_half_possession_output_0_or_1_np, threshold)
        return ret
    return objective_own_half_possession


if __name__ == "__main__":
    main()