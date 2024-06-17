import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def choice_data(competition_name, sequence_np, label_np, sequence_data, label_data, tactical_action_label, tactical_action_name_list, max_no_tactical_action, max_num_1_tactical_action):


    choice_sequence_list = []
    choice_label_list = []

    print(sequence_np.shape, label_np.shape)

    no_tactical_action = 0 # label が -1.0 の数
    num_1_tactical_action = 0 # label が 1.0 の数

    for i in range(label_np.shape[0]):

        # for j in range(label_np.shape[1]):

        if -1.0 < label_np[i, tactical_action_label] < 1.0:

            print(label_np[i, tactical_action_label])

            choice_sequence_list.append(sequence_np[i])
            choice_label_list.append(label_np[i, tactical_action_label])
        
        elif label_np[i, tactical_action_label] == 1.0:

            num_1_tactical_action += 1

            if no_tactical_action <= max_num_1_tactical_action:

                choice_sequence_list.append(sequence_np[i])
                choice_label_list.append(label_np[i, tactical_action_label])

        else:
            no_tactical_action += 1

            if no_tactical_action <= max_no_tactical_action:

                choice_sequence_list.append(sequence_np[i])
                choice_label_list.append(label_np[i, tactical_action_label])

    choice_sequence_np = np.stack(choice_sequence_list, axis=0)
    choice_label_np = np.stack(choice_label_list, axis=0)

    print(choice_sequence_np.shape, choice_label_np.shape)

    np.save("C:\\Users\\kento\\My_Research\\Data\\all_sequence_np\\" + sequence_data + "\\" + competition_name + "\\choice_" + tactical_action_name_list[tactical_action_label] + "_sequence.npy", choice_sequence_np)
    # np.save("C:\\Users\\kento\\My_Research\\Data\\all_label_np\\" + label_data + "\\" + competition_name + "\\choice_" + tactical_action_name_list[tactical_action_label] + "_label.npy", choice_label_np)