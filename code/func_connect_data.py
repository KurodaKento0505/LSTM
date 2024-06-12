import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
import numpy as np

def connect_data(competition_name, competition_id, season_id, data_length, grid_size_y, grid_size_x, sequence_data, label_data, purpose, num_player):

    # シーケンスとラベルを合体させたい

    # Set up the StatsBomb data loader
    SBL = StatsBombLoader()

    # Create a dataframe with all games from UEFA Euro
    df_games = SBL.games(competition_id, season_id).set_index("game_id")

    # 配列を格納するリスト
    data_arrays_list = []

    # sequence
    if purpose != 'label':
        if sequence_data == 'grid_' + str(grid_size_y) + '_' + str(grid_size_x):
            all_sequence_data_np = np.empty((0, data_length, grid_size_y * grid_size_x + 1))
        elif sequence_data == 'table_' + str(grid_size_y) + '_' + str(grid_size_x):
            all_sequence_data_np = np.empty((0, data_length, num_player * 2 + 3))
    # label
    if purpose != 'sequence':
        all_label_data_np = np.empty((0, 7))

    for i in range(len(df_games.index)):

        game_id = df_games.index[i]

        df_teams = SBL.teams(game_id)

        for j in range(len(df_teams.index)):

            print(i,j)

            main_team_id = df_teams.loc[j, 'team_id']

            if j == 0:
                if purpose != 'label':
                    sequence_data_np = np.load("C:\\Users\\kento\\My_Research\\Data\\comp_sequence_np\\" + sequence_data + "\\" + competition_name + "\\" + str(game_id) + "_1sthalf_" + str(main_team_id) + ".npy")
                if purpose != 'sequence':
                    label_data_np = np.load("C:\\Users\\kento\\My_Research\\Data\\comp_label_np\\" + label_data + "\\" + competition_name + "\\" + str(game_id) + "_1sthalf_" + str(main_team_id) + ".npy", allow_pickle=True)
            else:
                if purpose != 'label':
                    sequence_data_np = np.load("C:\\Users\\kento\\My_Research\\Data\\comp_sequence_np\\" + sequence_data + "\\" + competition_name + "\\" + str(game_id) + "_2ndhalf_" + str(main_team_id) + ".npy")
                if purpose != 'sequence':
                    label_data_np = np.load("C:\\Users\\kento\\My_Research\\Data\\comp_label_np\\" + label_data + "\\" + competition_name + "\\" + str(game_id) + "_2ndhalf_" + str(main_team_id) + ".npy", allow_pickle=True)
            
            if purpose != 'label':
                all_sequence_data_np = np.vstack((all_sequence_data_np, sequence_data_np))
            # label
            if purpose != 'sequence':
                all_label_data_np = np.vstack((all_label_data_np, label_data_np))
            
            # data_arrays_list.append(data_np)
            
    # all_data_np = np.concatenate(data_arrays_list, axis=0)
    
    if purpose != 'label':
        print('all_sequence_np.shape:', all_sequence_data_np.shape)
        np.save("C:\\Users\\kento\\My_Research\\Data\\all_sequence_np\\" + sequence_data + "\\" + competition_name + "\\all_sequence.npy", all_sequence_data_np)
        
    if purpose != 'sequence':
        print('all_label_np.shape:', all_label_data_np.shape)
        np.save("C:\\Users\\kento\\My_Research\\Data\\all_label_np\\" + label_data + "\\" + competition_name + "\\all_label.npy", all_label_data_np)
