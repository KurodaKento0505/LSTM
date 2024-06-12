import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


########################################################################################################################################

# data_length
data_length = 9

# grid_size
grid_size_x = 60
grid_size_y = 40

# channel
channel = 3

# data
label_data = 'time_to_tactical_action' # tactical_action, time_to_tactical_action

# test か否か
test = False

# val か否か
val = False

# make_graph か否か
make_graph = True
game_id = 3835331
main_team_id = 860

##########################################################################################################################################


if test:
    competition_name = "UEFA_Women's_Euro_2022" # FIFA_World_Cup_2022, UEFA_Euro_2020, UEFA_Women's_Euro_2022, Women's_World_Cup_2023
else:
    competition_name = "FIFA_World_Cup_2022"

print(competition_name)

# data
sequence_data = 'grid_' + str(grid_size_y) + '_' + str(grid_size_x)

if test:
    sequence_np = np.load("C:\\Users\\kento\\My_Research\\Data\\comp_sequence_np\\" + sequence_data + "\\" + competition_name + "\\"+ str(game_id) + "_1sthalf_" + str(main_team_id) + ".npy")
    label_np = np.load("C:\\Users\\kento\\My_Research\\Data\\comp_label_np\\" + label_data + "\\" + competition_name + "\\"+ str(game_id) + "_1sthalf_" + str(main_team_id) + ".npy", allow_pickle=True)
else:
    sequence_np = np.load("C:\\Users\\kento\\My_Research\\Data\\all_sequence_np\\" + sequence_data + "\\" + competition_name + "\\all_sequence.npy")
    label_np = np.load("C:\\Users\\kento\\My_Research\\Data\\all_label_np\\" + label_data + "\\" + competition_name + "\\all_label.npy", allow_pickle=True)

print(sequence_np.shape, label_np.shape)

for i in range(len(sequence_np)):

    for j in range(9):

        gray_image_np = np.zeros([grid_size_y, grid_size_x])

        for k in range(grid_size_x):

            for l in range(grid_size_y):

                gray_image_np[l, k] = sequence_np[i, j, grid_size_y * k + l + 1] * 50

        # img = np.squeeze(gray_image_np)
        plt.imshow(gray_image_np, cmap = "gray")
        plt.show()

    print(label_np[i])

    break