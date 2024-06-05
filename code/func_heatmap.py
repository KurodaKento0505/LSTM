import csv
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image


##########################################################################

# ガウス分布使うか否か
gause = False

# channel
channel = 3

##########################################################################


def make_heatmap(df, data_length, grid_size_x, grid_size_y):

    if channel == 1:
        # gray画像のシーケンスを作成する
        sequence_np = np.zeros((data_length, grid_size_y + 2, grid_size_x + 2), dtype=np.float64)
        
    elif channel == 3:
        # RGB画像のシーケンスを作成する
        sequence_np = np.zeros((data_length, grid_size_y + 2, grid_size_x + 2, 3), dtype=np.uint8)

    for i in range(data_length):

        # ガウス分布を使ってヒートマップの値を計算
        grid_ball_np = np.zeros((grid_size_y + 2, grid_size_x + 2))
        grid_teammate_np = np.zeros((grid_size_y + 2, grid_size_x + 2))
        grid_opponent_np = np.zeros((grid_size_y + 2, grid_size_x + 2))
        
        if channel == 1:
            # gray画像を作成する
            image_np = np.zeros((grid_size_y + 2, grid_size_x + 2), dtype=np.float64)
        elif channel == 3:
            # RGB画像を作成する
            image_np = np.zeros((grid_size_y + 2, grid_size_x + 2, 3), dtype=np.uint8)


        # 位置情報を格子状の行列に反映
        # ball
        grid_ball_x = int(round(df.loc[i, 'start_x'] / 120 / grid_size_x)) + 1
        grid_ball_y = int(round(df.loc[i, 'start_y'] / 80 / grid_size_y)) + 1

        if (grid_ball_x >= grid_size_x + 1) or (grid_ball_y >= grid_size_y + 1):
            continue

        else:
        
            # 格子状の行列に位置情報を反映 (人がいる場合は1)
            grid_ball_np[grid_ball_y, grid_ball_x] = 1.0

            if gause == True:
                grid_ball_np[grid_ball_y - 1, grid_ball_x] = 0.5
                grid_ball_np[grid_ball_y + 1, grid_ball_x] = 0.5
                grid_ball_np[grid_ball_y, grid_ball_x - 1] = 0.5
                grid_ball_np[grid_ball_y, grid_ball_x + 1] = 0.5
                grid_ball_np[grid_ball_y - 1, grid_ball_x - 1] = 0.25
                grid_ball_np[grid_ball_y - 1, grid_ball_x + 1] = 0.25
                grid_ball_np[grid_ball_y + 1, grid_ball_x - 1] = 0.25
                grid_ball_np[grid_ball_y + 1, grid_ball_x + 1] = 0.25

        # player
        # teammate
        for j in range(1,12):
            
            if df.loc[i, 'teammate_' + str(j) + '_x'] == -1.0:
                continue
            else:
                grid_teammate_x = int(round(df.loc[i, 'teammate_' + str(j) + '_x'] / 120 / grid_size_x)) + 1
                grid_teammate_y = int(round(df.loc[i, 'teammate_' + str(j) + '_y'] / 80 / grid_size_y)) + 1

                if (grid_teammate_x >= grid_size_x + 1) or (grid_teammate_y >= grid_size_y + 1):
                    continue

                else:

                    # 格子状の行列に位置情報を反映 (人がいる場合は1)
                    grid_teammate_np[grid_teammate_y, grid_teammate_x] = 1.0

                    if gause == True:
                        grid_teammate_np[grid_teammate_y - 1, grid_teammate_x] = 0.5
                        grid_teammate_np[grid_teammate_y + 1, grid_teammate_x] = 0.5
                        grid_teammate_np[grid_teammate_y, grid_teammate_x - 1] = 0.5
                        grid_teammate_np[grid_teammate_y, grid_teammate_x + 1] = 0.5
                        grid_teammate_np[grid_teammate_y - 1, grid_teammate_x - 1] = 0.25
                        grid_teammate_np[grid_teammate_y - 1, grid_teammate_x + 1] = 0.25
                        grid_teammate_np[grid_teammate_y + 1, grid_teammate_x - 1] = 0.25
                        grid_teammate_np[grid_teammate_y + 1, grid_teammate_x + 1] = 0.25

        # opponent
        for j in range(1,12):

            if df.loc[i, 'opponent_' + str(j) + '_x'] == -1.0:
                continue
            else:
                grid_opponent_x = int(round(df.loc[i, 'opponent_' + str(j) + '_x'] / 120 / grid_size_x)) + 1
                grid_opponent_y = int(round(df.loc[i, 'opponent_' + str(j) + '_y'] / 80 / grid_size_y)) + 1

                if (grid_opponent_x >= grid_size_x + 1) or (grid_opponent_y >= grid_size_y + 1):
                    continue

                else:

                    # 格子状の行列に位置情報を反映 (人がいる場合は1)
                    grid_opponent_np[grid_opponent_y, grid_opponent_x] = 1.0

                    if gause == True:
                        grid_opponent_np[grid_opponent_y - 1, grid_opponent_x] = 0.5
                        grid_opponent_np[grid_opponent_y + 1, grid_opponent_x] = 0.5
                        grid_opponent_np[grid_opponent_y, grid_opponent_x - 1] = 0.5
                        grid_opponent_np[grid_opponent_y, grid_opponent_x + 1] = 0.5
                        grid_opponent_np[grid_opponent_y - 1, grid_opponent_x - 1] = 0.25
                        grid_opponent_np[grid_opponent_y - 1, grid_opponent_x + 1] = 0.25
                        grid_opponent_np[grid_opponent_y + 1, grid_opponent_x - 1] = 0.25
                        grid_opponent_np[grid_opponent_y + 1, grid_opponent_x + 1] = 0.25

        if channel == 1:
            # グレースケール化
            image_np = grid_ball_np * 0.50 + grid_teammate_np * 0.25 + grid_opponent_np *  0.10

            # 時間情報の追加
            image_np[0, 0] = df.loc[i, 'time_reset']

            # gray画像のシーケンスを作成する
            sequence_np[i, :, :] = image_np
        
        elif channel == 3:

            image_np[:, :, 0] = grid_ball_np
            image_np[:, :, 1] = grid_teammate_np
            image_np[:, :, 2] = grid_opponent_np

            # 時間情報の追加
            image_np[0, 0, 0] = df.loc[i, 'time_reset']

            # gray画像のシーケンスを作成する
            sequence_np[i] = image_np
        
        # PILライブラリを使って画像を表示する
        # image = Image.fromarray(rgb_image_np)
        # image.save('rgb_image_' + str(i) + '.png')
        # cv2.imwrite('gray_image_' + str(i) + '.png', gray_image_np)
    
    return sequence_np


'''# 赤チャンネルにデータを代入
rgb_image_np[:, :, 0] = grid_ball_np * 255

# 緑チャンネルにデータを代入
rgb_image_np[:, :, 1] = grid_teammate_np * 255

# 青チャンネルにデータを代入
rgb_image_np[:, :, 2] = grid_opponent_np * 255

# RGB画像のシーケンスを作成する
rgb_image_sequence_np[i] = rgb_image_np'''