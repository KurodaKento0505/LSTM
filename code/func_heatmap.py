import csv
import pandas as pd
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image


##########################################################################

# ガウス分布使うか否か
gause = True

# channel
channel = 1

##########################################################################


def make_heatmap(df, data_length, grid_size_x, grid_size_y):

    if channel == 1:
        # gray画像のシーケンスを作成する
        sequence_np = np.zeros((data_length, grid_size_y * grid_size_x + 1), dtype=np.float64)
        
    elif channel == 3:
        # RGB画像のシーケンスを作成する
        sequence_np = np.zeros((data_length, grid_size_y, grid_size_x, 3), dtype=np.uint8)

    for i in range(data_length):
        
        if channel == 1:

            # ball, teammate, opponent
            # grid_ball_np = np.zeros(grid_size_y *  grid_size_x)
            # grid_teammate_np = np.zeros(grid_size_y *  grid_size_x)
            # grid_opponent_np = np.zeros(grid_size_y *  grid_size_x)

            # gray画像を作成する
            image_np = np.zeros(grid_size_y * grid_size_x + 1, dtype=np.float64)

        elif channel == 3:

            # ball, teammate, opponent
            grid_ball_np = np.zeros((grid_size_y, grid_size_x))
            grid_teammate_np = np.zeros((grid_size_y, grid_size_x))
            grid_opponent_np = np.zeros((grid_size_y, grid_size_x))

            # RGB画像を作成する
            image_np = np.zeros((grid_size_y, grid_size_x, 3), dtype=np.uint8)


        # 位置情報を格子状の行列に反映
        # ball
        ball_x = int(round(df.loc[i, 'start_x'] / (120 / grid_size_x)))
        ball_y = int(round(df.loc[i, 'start_y'] / (80 / grid_size_y)))

        # 外れ値
        if (ball_x >= grid_size_x) or (ball_y >= grid_size_y):
            continue

        else:
        
            # 格子状の行列に位置情報を反映 (人がいる場合は1)
            image_np[grid_size_y * ball_x + ball_y + 1] += 5.0

            if gause == True:
                grid_ball_np[ball_y - 1, ball_x] = 0.5
                grid_ball_np[ball_y + 1, ball_x] = 0.5
                grid_ball_np[ball_y, ball_x - 1] = 0.5
                grid_ball_np[ball_y, ball_x + 1] = 0.5
                grid_ball_np[ball_y - 1, ball_x - 1] = 0.25
                grid_ball_np[ball_y - 1, ball_x + 1] = 0.25
                grid_ball_np[ball_y + 1, ball_x - 1] = 0.25
                grid_ball_np[ball_y + 1, ball_x + 1] = 0.25

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
                    image_np[grid_size_y * teammate_x + teammate_y + 1] += 3.0

                    if gause == True:
                        grid_teammate_np[teammate_y - 1, teammate_x] = 0.5
                        grid_teammate_np[teammate_y + 1, teammate_x] = 0.5
                        grid_teammate_np[teammate_y, teammate_x - 1] = 0.5
                        grid_teammate_np[teammate_y, teammate_x + 1] = 0.5
                        grid_teammate_np[teammate_y - 1, teammate_x - 1] = 0.25
                        grid_teammate_np[teammate_y - 1, teammate_x + 1] = 0.25
                        grid_teammate_np[teammate_y + 1, teammate_x - 1] = 0.25
                        grid_teammate_np[teammate_y + 1, teammate_x + 1] = 0.25

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
                    image_np[grid_size_y * opponent_x + opponent_y + 1] += 1.0

                    if gause == True:
                        grid_opponent_np[opponent_y - 1, opponent_x] = 0.5
                        grid_opponent_np[opponent_y + 1, opponent_x] = 0.5
                        grid_opponent_np[opponent_y, opponent_x - 1] = 0.5
                        grid_opponent_np[opponent_y, opponent_x + 1] = 0.5
                        grid_opponent_np[opponent_y - 1, opponent_x - 1] = 0.25
                        grid_opponent_np[opponent_y - 1, opponent_x + 1] = 0.25
                        grid_opponent_np[opponent_y + 1, opponent_x - 1] = 0.25
                        grid_opponent_np[opponent_y + 1, opponent_x + 1] = 0.25

        if channel == 1:
            # グレースケール化
            # image_np = grid_ball_np * 5 + grid_teammate_np * 3 + grid_opponent_np *  1

            # 時間情報の追加
            image_np[0] = df.loc[i, 'time_reset'] / 10

            # gray画像のシーケンスを作成する
            sequence_np[i, :] = image_np
        
        elif channel == 3:

            image_np[:, :, 0] = grid_ball_np
            image_np[:, :, 1] = grid_teammate_np
            image_np[:, :, 2] = grid_opponent_np

            # 時間情報の追加
            image_np[0, 0, 0] = df.loc[i, 'time_reset']

            # gray画像のシーケンスを作成する
            sequence_np[i] = image_np
    
    return sequence_np



def make_heatmap_from_table(sequence_np, new_grid_size_x, new_grid_size_y):

    new_sequence_np = np.zeros((sequence_np.shape[0], sequence_np.shape[1], 3 + ((new_grid_size_y + 2) * (new_grid_size_x + 2)) * 2))

    for i in range(sequence_np.shape[0]):

        for j in range(sequence_np.shape[1]):

            grid_teammate_np = np.zeros((new_grid_size_y + 2, new_grid_size_x + 2))
            grid_opponent_np = np.zeros((new_grid_size_y + 2, new_grid_size_x + 2))

            for k in range(sequence_np.shape[2]):

                # time_seconds
                if k == 0:
                    new_sequence_np[i, j, k] = sequence_np[i, j, k]
                # ball_x, ball_y
                elif k < 3:
                    new_sequence_np[i, j, k] = int(round(sequence_np[i, j, k] / (60 / new_grid_size_x)))

                # teammate
                elif 3 <= k <= 24:

                    # x座標
                    if k % 2 == 1:

                        # 外れ値
                        if sequence_np[i, j, k] == -1.0:
                            continue

                        teammate_x = int(round(sequence_np[i, j, k] / (60 / new_grid_size_x)))
                    
                    # y座標
                    else:

                        # 外れ値
                        if sequence_np[i, j, k] == -1.0:
                            continue

                        teammate_y = int(round(sequence_np[i, j, k] / (60 / new_grid_size_x)))

                        if gause == True:
                            grid_teammate_np[teammate_y, teammate_x] = 1.0
                            grid_teammate_np[teammate_y - 1, teammate_x] = 0.5
                            grid_teammate_np[teammate_y + 1, teammate_x] = 0.5
                            grid_teammate_np[teammate_y, teammate_x - 1] = 0.5
                            grid_teammate_np[teammate_y, teammate_x + 1] = 0.5
                            grid_teammate_np[teammate_y - 1, teammate_x - 1] = 0.25
                            grid_teammate_np[teammate_y - 1, teammate_x + 1] = 0.25
                            grid_teammate_np[teammate_y + 1, teammate_x - 1] = 0.25
                            grid_teammate_np[teammate_y + 1, teammate_x + 1] = 0.25
                        else:
                            grid_teammate_np[teammate_y, teammate_x] = 1.0

                # opponent
                else:
                    
                    # x座標
                    if k % 2 == 1:

                        # 外れ値
                        if sequence_np[i, j, k] == -1.0:
                            continue

                        opponent_x = int(round(sequence_np[i, j, k] / (60 / new_grid_size_x)))
                    
                    # y座標
                    else:

                        # 外れ値
                        if sequence_np[i, j, k] == -1.0:
                            continue

                        opponent_y = int(round(sequence_np[i, j, k] / (60 / new_grid_size_x)))

                        if gause == True:
                            grid_opponent_np[opponent_y, opponent_x] = 1.0
                            grid_opponent_np[opponent_y - 1, opponent_x] = 0.5
                            grid_opponent_np[opponent_y + 1, opponent_x] = 0.5
                            grid_opponent_np[opponent_y, opponent_x - 1] = 0.5
                            grid_opponent_np[opponent_y, opponent_x + 1] = 0.5
                            grid_opponent_np[opponent_y - 1, opponent_x - 1] = 0.25
                            grid_opponent_np[opponent_y - 1, opponent_x + 1] = 0.25
                            grid_opponent_np[opponent_y + 1, opponent_x - 1] = 0.25
                            grid_opponent_np[opponent_y + 1, opponent_x + 1] = 0.25
                        else:
                            grid_opponent_np[opponent_y, opponent_x] = 1.0

            '''grid_np = grid_teammate_np + grid_opponent_np
            plt.imshow(grid_np, cmap = "gray")
            plt.show()'''
            
            # くっつける
            for l in range(new_grid_size_x + 2):

                for m in range(new_grid_size_y + 2):

                    # teammate
                    new_sequence_np[i, j, l * (new_grid_size_y + 2) + m + 3] = grid_teammate_np[m, l]

                    # opponent
                    new_sequence_np[i, j, l * (new_grid_size_y + 2) + m + 3 + (new_grid_size_y + 2) * (new_grid_size_x + 2)] = grid_opponent_np[m, l]

        if i % 100 == 0:
            print(i)

    return new_sequence_np



'''# PILライブラリを使って画像を表示する
        gray_image_np = np.zeros([grid_size_y, grid_size_x])
        for k in range(grid_size_x):
            for l in range(grid_size_y):
                gray_image_np[l, k] = image_np[grid_size_y * k + l + 1] * 50
        plt.imshow(gray_image_np, cmap = "gray")
        plt.show()'''


'''# 赤チャンネルにデータを代入
rgb_image_np[:, :, 0] = grid_ball_np * 255

# 緑チャンネルにデータを代入
rgb_image_np[:, :, 1] = grid_teammate_np * 255

# 青チャンネルにデータを代入
rgb_image_np[:, :, 2] = grid_opponent_np * 255

# RGB画像のシーケンスを作成する
rgb_image_sequence_np[i] = rgb_image_np'''