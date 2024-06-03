import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sequence_np = np.load("C:\\Users\\kento\\OneDrive\\My_Research\\Data\\comp_sequence_np\\FIFA_World_Cup_2022\\3857254_1sthalf_776.npy")
label_np = np.load("C:\\Users\\kento\\OneDrive\\My_Research\\Data\\comp_label_np\\FIFA_World_Cup_2022\\3857254_1sthalf_776.npy", allow_pickle=True)

print(sequence_np.shape, label_np.shape)

for i in range(len(sequence_np)):

    for j in range(9):

        plt.imshow(sequence_np[i, j], cmap = "gray")
        plt.show()

    print(label_np[i])

    break