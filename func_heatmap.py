import csv
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image


def make_heatmap(df, data_length):

    for i in range(data_length):

        # 格子のサイズ
        grid_size_x = 113  # x座標の最大値 + 1
        grid_size_y = 73   # y座標の最大値 + 1

        # ガウス分布を使ってヒートマップの値を計算
        grid_ball_np = np.zeros((grid_size_y, grid_size_x))
        grid_teammate_np = np.zeros((grid_size_y, grid_size_x))
        grid_opponent_np = np.zeros((grid_size_y, grid_size_x))
        
        # RGB画像を作成する
        rgb_image_np = np.zeros((grid_size_y, grid_size_x, 3), dtype=np.uint8)

        # RGB画像のシーケンスを作成する
        rgb_image_sequence_np = np.zeros((grid_size_y, grid_size_x, 3, data_length), dtype=np.uint8)

        # ガウス分布のパラメータ
        mean = [0, 0]  # 平均
        covariance = [[25, 0], [0, 25]]  # 共分散行列

        # 位置情報を格子状の行列に反映
        # ball
        grid_ball_x = int(round(df.loc[i, 'start_x'] * 110)) + 1
        grid_ball_y = int(round(df.loc[i, 'start_y'] * 70)) + 1
        # 格子状の行列に位置情報を反映 (人がいる場合は1)
        grid_ball_np[grid_ball_y, grid_ball_x] = 1.0
        grid_ball_np[grid_ball_y - 1, grid_ball_x] = 0.5
        grid_ball_np[grid_ball_y + 1, grid_ball_x] = 0.5
        grid_ball_np[grid_ball_y, grid_ball_x - 1] = 0.5
        grid_ball_np[grid_ball_y, grid_ball_x + 1] = 0.5
        grid_ball_np[grid_ball_y - 1, grid_ball_x - 1] = 0.25
        grid_ball_np[grid_ball_y - 1, grid_ball_x + 1] = 0.25
        grid_ball_np[grid_ball_y + 1, grid_ball_x - 1] = 0.25
        grid_ball_np[grid_ball_y + 1, grid_ball_x + 1] = 0.25

        # player
        for j in range(1,12):
            
            # teammate
            if df.loc[i, 'teammate_' + str(j) + '_x'] == -1.0:
                pass
            else:
                grid_teammate_x = int(round(df.loc[i, 'teammate_' + str(j) + '_x'] * 110)) + 1
                grid_teammate_y = int(round(df.loc[i, 'teammate_' + str(j) + '_y'] * 70)) + 1

                # 格子状の行列に位置情報を反映 (人がいる場合は1)
                grid_teammate_np[grid_teammate_y, grid_teammate_x] = 1.0
                grid_teammate_np[grid_teammate_y - 1, grid_teammate_x] = 0.5
                grid_teammate_np[grid_teammate_y + 1, grid_teammate_x] = 0.5
                grid_teammate_np[grid_teammate_y, grid_teammate_x - 1] = 0.5
                grid_teammate_np[grid_teammate_y, grid_teammate_x + 1] = 0.5
                grid_teammate_np[grid_teammate_y - 1, grid_teammate_x - 1] = 0.25
                grid_teammate_np[grid_teammate_y - 1, grid_teammate_x + 1] = 0.25
                grid_teammate_np[grid_teammate_y + 1, grid_teammate_x - 1] = 0.25
                grid_teammate_np[grid_teammate_y + 1, grid_teammate_x + 1] = 0.25

            # opponent
            if df.loc[i, 'opponent_' + str(j) + '_x'] == -1.0:
                pass
            else:
                grid_opponent_x = int(round(df.loc[i, 'opponent_' + str(j) + '_x'] * 110)) + 1
                grid_opponent_y = int(round(df.loc[i, 'opponent_' + str(j) + '_y'] * 70)) + 1

                # 格子状の行列に位置情報を反映 (人がいる場合は1)
                grid_opponent_np[grid_opponent_y, grid_opponent_x] = 1.0
                grid_opponent_np[grid_opponent_y - 1, grid_opponent_x] = 0.5
                grid_opponent_np[grid_opponent_y + 1, grid_opponent_x] = 0.5
                grid_opponent_np[grid_opponent_y, grid_opponent_x - 1] = 0.5
                grid_opponent_np[grid_opponent_y, grid_opponent_x + 1] = 0.5
                grid_opponent_np[grid_opponent_y - 1, grid_opponent_x - 1] = 0.25
                grid_opponent_np[grid_opponent_y - 1, grid_opponent_x + 1] = 0.25
                grid_opponent_np[grid_opponent_y + 1, grid_opponent_x - 1] = 0.25
                grid_opponent_np[grid_opponent_y + 1, grid_opponent_x + 1] = 0.25

        # 赤チャンネルにデータを代入
        rgb_image_np[:, :, 0] = grid_ball_np * 255

        # 緑チャンネルにデータを代入
        rgb_image_np[:, :, 1] = grid_teammate_np * 255

        # 青チャンネルにデータを代入
        rgb_image_np[:, :, 2] = grid_opponent_np * 255

        # PILライブラリを使って画像を表示する
        # image = Image.fromarray(rgb_image_np)
        # image.show()
        
        # RGB画像のシーケンスを作成する
        rgb_image_sequence_np[:, :, :, i] = rgb_image_np

    return rgb_image_sequence_np