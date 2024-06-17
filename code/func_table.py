import csv
import pandas as pd
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image



def make_table(df, data_length, grid_size_x, grid_size_y, num_player):


    # gray画像のシーケンスを作成する
    sequence_np = np.zeros((data_length, num_player * 2 + 3), dtype=np.float64)


    for i in range(data_length):


        # gray画像を作成する
        event_np = np.full(num_player * 2 + 3, -1.0, dtype=np.float64)

        # 時間情報の追加
        event_np[0] = df.loc[i, 'time_reset']


        # 位置情報を格子状の行列に反映
        # ball
        ball_x = int(round(df.loc[i, 'start_x'] / (120 / grid_size_x)))
        ball_y = int(round(df.loc[i, 'start_y'] / (80 / grid_size_y)))

        # 外れ値
        if (ball_x >= grid_size_x) or (ball_y >= grid_size_y):
            continue

        else:
        
            # 格子状の行列に位置情報を反映 (人がいる場合は1)
            event_np[1] = ball_x
            event_np[2] = ball_y


        # player
        # teammate
        for j in range(1,12):
            
            if df.loc[i, 'teammate_' + str(j) + '_x'] == -1.0:
                continue
            else:
                teammate_x = int(round(df.loc[i, 'teammate_' + str(j) + '_x'] / (120 / grid_size_x)))
                teammate_y = int(round(df.loc[i, 'teammate_' + str(j) + '_y'] / (80 / grid_size_y)))

                if (teammate_x >= grid_size_x) or (teammate_y >= grid_size_y):
                    continue

                else:

                    # 格子状の行列に位置情報を反映 (人がいる場合は1)
                    event_np[j * 2 + 1] = teammate_x
                    event_np[j * 2 + 2] = teammate_y


        # opponent
        for j in range(1,12):

            if df.loc[i, 'opponent_' + str(j) + '_x'] == -1.0:
                continue
            else:
                opponent_x = int(round(df.loc[i, 'opponent_' + str(j) + '_x'] / (120 / grid_size_x)))
                opponent_y = int(round(df.loc[i, 'opponent_' + str(j) + '_y'] / (80 / grid_size_y)))

                if (opponent_x >= grid_size_x) or (opponent_y >= grid_size_y):
                    continue

                else:

                    # 格子状の行列に位置情報を反映 (人がいる場合は1)
                    event_np[j * 2 + 23] = opponent_x
                    event_np[j * 2 + 24] = opponent_y


        # gray画像のシーケンスを作成する
        sequence_np[i] = event_np


    return sequence_np
