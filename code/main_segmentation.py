import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import matplotsoccer as mps

from func_label_tactical_actions import label_tactical_actions
from func_arrange_data import arrange_data
from func_heatmap import make_heatmap
from func_table import make_table
from func_connect_data import connect_data
from func_label_time import label_time_by_starting
from func_label_attack_or_defense import label_attack_or_defense


# Set up the StatsBomb data loader
SBL = StatsBombLoader()

# View all available competitions
df_competitions = SBL.competitions()


##################################################################################################################################################
competition_id = 53
season_id = 106
competition_name = "UEFA_Women's_Euro_2022" # FIFA_World_Cup_2022(43,106), UEFA_Euro_2020(55,43), UEFA_Women's_Euro_2022(53,106), Women's_World_Cup_2023

# connect
connect = False

# sequence, label or both
purpose = 'label'

# kind_of_sequence
kind_of_sequence = 'table'

# heatmap
grid_size_x = 60
grid_size_y = 40

# the number of players
num_player = 22

# data_length
data_length = 9

label_data = 'time_to_tactical_action' # tactical_action, time_to_tactical_action
###################################################################################################################################################


# data
if kind_of_sequence == 'grid':

    sequence_data = 'grid_' + str(grid_size_y) + '_' + str(grid_size_x)

elif kind_of_sequence == 'table':

    sequence_data = 'table_' + str(grid_size_y) + '_' + str(grid_size_x)


# Create a dataframe with all games from UEFA Euro
df_games = SBL.games(competition_id, season_id).set_index("game_id")

# 各戦術的行動の名前 
if label_data != 'attack_or_defense':
    tactical_action_name_list = ['longcounter', 'shortcounter', 'opposition_half_possession', 'own_half_possession', 'counterpressing', 'highpressing', 'middlepressing']
else:
    tactical_action_name_list = ['attack', 'defense']

# 戦術的行動の数
num_tactical_action = len(tactical_action_name_list)


'''# 一つの大会におけるcount sequence
count_sequence = 0

# 一つの大会における戦術的行動の数
include_longcounter = 0
include_shortcounter = 0
include_opposition_half_possession = 0
include_own_half_possession = 0
include_counterpressing = 0
include_highpressing = 0
include_middlepressing = 0

# in playの長さ
in_play_np = np.empty([0])'''


def main():



    # segmentation for making sequence and label dataset
    if connect != True:

        for i in range(len(df_games.index)): # reversed

            '''if i == 0 or i == 1 or i == 2:
                continue'''
            
            '''if i == len(df_games.index) - 1: # or i == len(df_games.index) - 2 or i == len(df_games.index) - 3:
                continue'''

            game_id = df_games.index[i]
            df_teams = SBL.teams(game_id)
            df_players = SBL.players(game_id)
            df_events = SBL.events(game_id, True)  # 360dataが含まれる = True

            home_team_id = df_games.at[game_id, "home_team_id"]


            # spdlによって、df_eventsをdf_actionsに変更
            df_actions = spadl.statsbomb.convert_to_actions(df_events, home_team_id)

            # Replace result, actiontype and bodypart IDs by their corresponding name
            df_actions = spadl.add_names(df_actions)

            # Add team and player names
            df_actions = df_actions.merge(df_teams).merge(df_players)
            
            # period_id,timestampで並べ替え
            df_actions = df_actions.sort_values(['period_id','time_seconds']) 


            # データを選別、'360_data'、'possession'、'player_name'
            df_actions = df_actions.loc[:,['period_id','time_seconds','play_pattern_name','team_name','team_id','position_name','position_id','type_name','result_name','under_pressure','counterpress','360_data','start_x','start_y','end_x','end_y']]


            # 前後半で判別
            df_actions_1sthalf = df_actions[df_actions["period_id"] == 1]
            df_actions_2ndhalf = df_actions[df_actions["period_id"] == 2]

            # indexの振り直し
            df_actions_1sthalf.reset_index(drop=True, inplace=True)
            df_actions_2ndhalf.reset_index(drop=True, inplace=True)

            # 2つのチームで main_team_id をループ
            for j in range(2):

                # 戦術的行動のラベルを割り振る場所を作成
                for k in range(num_tactical_action):
                    df_actions_1sthalf[tactical_action_name_list[k]] = 0
                    df_actions_2ndhalf[tactical_action_name_list[k]] = 0
                
                # 戦術的行動の開始時刻までの時間を割り振る場所を作成
                for k in range(num_tactical_action):
                    df_actions_1sthalf['time_to_' + tactical_action_name_list[k]] = -1.0
                    df_actions_2ndhalf['time_to_' + tactical_action_name_list[k]] = -1.0
                
                # ラベルがついているイベントデータにしるしをつけて，結果を比較しやすくする
                df_actions_1sthalf['label'] = False
                df_actions_2ndhalf['label'] = False

                main_team_id = df_teams.loc[j, 'team_id'].copy()

                # main_team_id 入力
                df_actions_1sthalf.loc[0, 'main_team_id'] = main_team_id
                df_actions_2ndhalf.loc[0, 'main_team_id'] = main_team_id

                print('main_team_id:',main_team_id)

                arrange_data(df_actions_1sthalf, main_team_id)
                arrange_data(df_actions_2ndhalf, main_team_id)

                label_and_sequence_split(df_actions_1sthalf, main_team_id, game_id, 1)
                label_and_sequence_split(df_actions_2ndhalf, main_team_id, game_id, 2)

                # df_actions to csv
                # df_actions_1sthalf.to_csv("C:\\Users\\kento\\My_Research\\Data\\df_actions\\" + label_data + "\\" + competition_name + "\\"+ str(game_id) + "_1sthalf_" + str(main_team_id) + ".csv")
                # df_actions_2ndhalf.to_csv("C:\\Users\\kento\\My_Research\\Data\\df_actions\\" + label_data + "\\" + competition_name + "\\"+ str(game_id) + "_2ndhalf_" + str(main_team_id) + ".csv")

                # df_actions_1sthalf.drop(['360_data'], axis=1)


            print((i + 1) / len(df_games))
            break
    
    # connect sequence_data
    else:
        connect_data(competition_name, competition_id, season_id, data_length, grid_size_y, grid_size_x, sequence_data, label_data, purpose, num_player)




# 攻撃シーケンス分類かつそれぞれの局面関数に移動させてラベル付与
def label_and_sequence_split(df_actions_half, main_team_id, game_id, half):


    next_in_play_start = 0

    # sequence
    comp_sequence_np = np.empty((0, data_length, num_player * 2 + 3))
    # sequence_np_list = []

    # label
    # 戦術的行動
    comp_label_np = np.empty((0, 7))
    # 攻守
    # comp_label_np = np.empty((0, 2))

    for i in range(len(df_actions_half) - 1):


        # １．out of playを区切る

        # NaNの対処
        if (df_actions_half.loc[i, ['play_pattern_name']].isnull() == True).any().any():
            pass
        elif (df_actions_half.loc[i + 1, ['play_pattern_name']].isnull() == True).any().any():
            pass

        # play_pattern_nameが変わったらout of play，前のイベントと20秒以上離れていたら区切る
        elif (df_actions_half.loc[i, ['play_pattern_name']] != df_actions_half.loc[i + 1, ['play_pattern_name']]).any().any() or (int(df_actions_half.loc[i + 1, ['time_seconds']].iloc[0]) - int(df_actions_half.loc[i, ['time_seconds']].iloc[0]) >= 15) or (df_actions_half.loc[i + 1, ['type_name']] == 'throw_in').any().any() or (df_actions_half.loc[i + 1, ['type_name']] == 'corner_kick').any().any() or (df_actions_half.loc[i + 1, ['type_name']] == 'goalkick').any().any() or (df_actions_half.loc[i + 1, ['type_name']] == 'freekick_crossed').any().any():

            # play_patternで区切った場合ややこしい
            if (df_actions_half.loc[i, ['play_pattern_name']] != df_actions_half.loc[i + 1, ['play_pattern_name']]).any().any():
            
                # 次の play_pattern_name が regular play or From Counter の場合、継続
                if (df_actions_half.loc[i + 1, ['play_pattern_name']].isin(['Regular Play']) == True).any().any() or (df_actions_half.loc[i + 1, ['play_pattern_name']].isin(['From Counter']) == True).any().any():
                    continue


            df_in_play = df_actions_half[next_in_play_start : i + 1]

            # indexの振り直し
            df_in_play.reset_index(drop=True, inplace=True)

            # data_length より短い df_in_play は使えない
            if len(df_in_play) <= data_length:
                next_in_play_start += len(df_in_play)
                continue


            if purpose != 'sequence':
                
                # ラベル付け
                # 戦術的行動のラベル付け
                if label_data == 'tactical_action':
                    label_tactical_actions(df_in_play, main_team_id)

                # 戦術的行動の開始時刻までの時間距離のラベル付け 
                elif label_data == 'time_to_tactical_action':
                    label_tactical_actions(df_in_play, main_team_id)
                    label_time_by_starting(df_in_play, main_team_id, tactical_action_name_list)

                # 攻守のラベル付け
                else:
                    label_attack_or_defense(df_in_play, main_team_id)


            # シーケンス分割
            # 注目データと前9データを一つのシーケンスに
            for j in range(data_length, len(df_in_play)):

                if purpose != 'label':

                    # sequence分割
                    df_sequence = df_in_play.loc[ j - data_length : j , ['time_seconds','start_x','start_y','teammate_count','opponent_count','teammate_1_x','teammate_1_y','opponent_1_x','opponent_1_y','teammate_2_x','teammate_2_y','opponent_2_x','opponent_2_y','teammate_3_x','teammate_3_y','opponent_3_x','opponent_3_y','teammate_4_x','teammate_4_y','opponent_4_x','opponent_4_y','teammate_5_x','teammate_5_y','opponent_5_x','opponent_5_y','teammate_6_x','teammate_6_y','opponent_6_x','opponent_6_y','teammate_7_x','teammate_7_y','opponent_7_x','opponent_7_y','teammate_8_x','teammate_8_y','opponent_8_x','opponent_8_y','teammate_9_x','teammate_9_y','opponent_9_x','opponent_9_y','teammate_10_x','teammate_10_y','opponent_10_x','opponent_10_y','teammate_11_x','teammate_11_y','opponent_11_x','opponent_11_y']].copy()

                    # indexの振り直し
                    df_sequence.reset_index(drop=True, inplace=True)

                    # time_reset の追加
                    time_reset(df_sequence)
                    
                    # from sequence to heatmap or table
                    # sequence_np が返ってくる
                    # sequence_np = make_heatmap(df_sequence, data_length, grid_size_x, grid_size_y)
                    sequence_np = make_table(df_sequence, data_length, grid_size_x, grid_size_y, num_player)

                    # シーケンスをまとめる
                    # sequence_np_list.append(sequence_np)
                    comp_sequence_np = np.vstack((comp_sequence_np, [sequence_np]))


                if purpose != 'sequence':

                    # label分割
                    # 戦術的行動
                    if label_data == 'tactical_action':
                        label_np = df_in_play.loc[j, ['longcounter','shortcounter','opposition_half_possession','own_half_possession','counterpressing','highpressing','middlepressing']].to_numpy()
                    # 戦術的行動の開始時刻までの時間
                    elif label_data == 'time_to_tactical_action':
                        label_np = df_in_play.loc[j, ['time_to_longcounter','time_to_shortcounter','time_to_opposition_half_possession','time_to_own_half_possession','time_to_counterpressing','time_to_highpressing','time_to_middlepressing']].to_numpy()
                    # 攻守
                    else:
                        label_np = df_in_play.loc[j, ['attack', 'defense']].to_numpy()

                    # ラベルがついているイベントデータにしるしをつけて，結果を比較しやすくする
                    df_in_play.loc[j, 'label'] = True

                    # ラベルをまとめる
                    comp_label_np = np.vstack((comp_label_np, [label_np]))

                # count_sequence
                # count_sequence += 1

            next_in_play_start = i + 1

    # comp_sequence_np = np.concatenate(sequence_np_list, axis=0)

    # print(comp_sequence_np.shape)
    
    # save sequence_np
    # 前半
    if half == 1:
        if purpose != 'label':
            np.save('C:\\Users\kento\\My_Research\Data\\comp_sequence_np\\' + sequence_data + '\\' + competition_name + '\\'+ str(game_id) + '_1sthalf_' + str(main_team_id), comp_sequence_np)
        if purpose != 'sequence':
            np.save('C:\\Users\kento\\My_Research\Data\\comp_label_np\\' + label_data + '\\' + competition_name + '\\'+ str(game_id) + '_1sthalf_' + str(main_team_id), comp_label_np)
    # 後半
    else:
        if purpose != 'label':
            np.save('C:\\Users\kento\\My_Research\Data\\comp_sequence_np\\' + sequence_data + '\\' + competition_name + '\\'+ str(game_id) + '_2ndhalf_' + str(main_team_id), comp_sequence_np)
        if purpose != 'sequence':
            np.save('C:\\Users\kento\\My_Research\Data\\comp_label_np\\' + label_data + '\\' + competition_name + '\\'+ str(game_id) + '_2ndhalf_' + str(main_team_id), comp_label_np)
    
    # 戦術的行動の数
    # print(include_longcounter,include_shortcounter,include_opposition_half_possession,include_own_half_possession,include_counterpressing,include_highpressing,include_middlepressing)





# longcounterとshortcounterの長さ
def counter_length(df_actions_half):

    global longcounter_length_np
    global shortcounter_length_np

    end_counter = 0
    
    for i in range(len(df_actions_half)):

        if i < end_counter:
            continue

        # longcounterの長さ
        if (df_actions_half.loc[i, ['label']] == 1).any().any():

            for j in range(i, len(df_actions_half)):
                if (df_actions_half.loc[j, ['label']] != 1).any().any():

                    # longcounterの長さ
                    longcounter_length = j - i
                    
                    # label_lengthに格納
                    df_actions_half.loc[i : j - 1, 'label_length'] = longcounter_length

                    # numpyに格納
                    longcounter_length_np = np.append(longcounter_length_np, longcounter_length)
                    end_counter = j
                    break


        # shortcounterの長さ
        elif (df_actions_half.loc[i, ['label']] == 2).any().any():

            for j in range(i, len(df_actions_half)):
                if (df_actions_half.loc[j, ['label']] != 2).any().any():
                    
                    # shortcounterの長さ
                    shortcounter_length = j - i
                    
                    # label_lengthに格納
                    df_actions_half.loc[i : j - 1, 'label_length'] = shortcounter_length

                    # numpyに格納
                    shortcounter_length_np = np.append(shortcounter_length_np, shortcounter_length)
                    end_counter = j
                    break


# データの正規化
def normalization(df):
    df.loc[:,['start_x']] = df.loc[:,['start_x']] / 120
    df.loc[:,['start_y']] = df.loc[:,['start_y']] / 80
    df.loc[:,['time_reset']] = df.loc[:,['time_reset']] / 100

    for i in range(len(df)):
        for n in range(1, 12):
            if df.at[i, 'teammate_' + str(n) + '_x'] == -1.0:
                pass
            else:
                df.at[i, 'teammate_' + str(n) + '_x'] = df.at[i, 'teammate_' + str(n) + '_x'] / 120
                df.at[i, 'teammate_' + str(n) + '_y'] = df.at[i, 'teammate_' + str(n) + '_y'] / 80

            if df.at[i, 'opponent_' + str(n) + '_x'] == -1.0:
                pass
            else:
                df.at[i, 'opponent_' + str(n) + '_x'] = df.at[i, 'opponent_' + str(n) + '_x'] / 120
                df.at[i, 'opponent_' + str(n) + '_y'] = df.at[i, 'opponent_' + str(n) + '_y'] / 80
    return df


# time_seconds を整形
def time_reset(df):

    for i in range(len(df - 1)):
        if i == 0:
            first_time_seconds = float(df.loc[i, ['time_seconds']].iloc[0])
            df.loc[i, ['time_reset']] = 0
        else:
            df.loc[i, ['time_reset']] = float(df.loc[i, ['time_seconds']].iloc[0]) - first_time_seconds


# それぞれのシーケンスをto_csv
def sequence_to_csv(df_sequence):

    global competition_name
    global count_sequence
    global convert

    global include_longcounter
    global include_shortcounter
    global include_opposition_half_possession
    global include_own_half_possession
    global include_counterpressing
    global include_highpressing
    global include_middlepressing


    df_sequence = time_reset(df_sequence)
    df_sequence = normalization(df_sequence)

    if convert:
        count_sequence += 1
        # df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\test_sequence\\" + competition_name + "_convert\\" + str(count_sequence).zfill(6) + ".csv")
        print(str(count_sequence).zfill(6))

        '''# if (df_sequence.loc[0, ['until_longcounter']] != 1).any().any():
        if sequence_type == 1:
            count_sequence += 1
            include_longcounter += 1
            df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\longcounter\\" + str(include_longcounter).zfill(6) + ".csv")
            print(str(include_longcounter).zfill(6))
        
        # elif (df_sequence.loc[0, ['until_shortcounter']] != 1).any().any():
        elif sequence_type == 2:
            count_sequence += 1
            include_shortcounter += 1
            df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\shortcounter\\" + str(include_shortcounter).zfill(6) + ".csv")
            print(str(include_shortcounter).zfill(6))

        elif (df_sequence.loc[0, ['until_opposition_half_possession']] != 1).any().any():
            count_sequence += 1
            include_opposition_half_possession += 1
            df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\opposition_half_possession\\" + str(include_opposition_half_possession).zfill(6) + ".csv")
            print(str(include_opposition_half_possession).zfill(6))

        elif (df_sequence.loc[0, ['until_own_half_possession']] != 1).any().any():
            count_sequence += 1
            include_own_half_possession += 1
            df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\own_half_possession\\" + str(include_own_half_possession).zfill(6) + ".csv")
            print(str(include_own_half_possession).zfill(6))

        else:
            count_sequence += 1
            no_tactical_action += 1
            if no_tactical_action <= 10000:
                df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\others\\" + str(no_tactical_action).zfill(6) + ".csv")
                print(str(no_tactical_action).zfill(6))'''

    else:
        count_sequence += 1
        # df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\test_sequence\\" + competition_name + "\\" + str(count_sequence).zfill(6) + ".csv")
        print(str(count_sequence).zfill(6))

        '''# if (df_sequence.loc[0, ['until_longcounter']] != 1).any().any():
        if sequence_type == 1:
            count_sequence += 1
            include_longcounter += 1
            df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\longcounter\\" + str(include_longcounter).zfill(6) + ".csv")
            print(str(include_longcounter).zfill(6))
        
        # elif (df_sequence.loc[0, ['until_shortcounter']] != 1).any().any():
        elif sequence_type == 2:
            count_sequence += 1
            include_shortcounter += 1
            df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\shortcounter\\" + str(include_shortcounter).zfill(6) + ".csv")
            print(str(include_shortcounter).zfill(6))

        elif (df_sequence.loc[0, ['until_opposition_half_possession']] != 1).any().any():
            count_sequence += 1
            include_opposition_half_possession += 1
            df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\opposition_half_possession\\" + str(include_opposition_half_possession).zfill(6) + ".csv")
            print(str(include_opposition_half_possession).zfill(6))

        elif (df_sequence.loc[0, ['until_own_half_possession']] != 1).any().any():
            count_sequence += 1
            include_own_half_possession += 1
            df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\own_half_possession\\" + str(include_own_half_possession).zfill(6) + ".csv")
            print(str(include_own_half_possession).zfill(6))

        else:
            count_sequence += 1
            no_tactical_action += 1
            if no_tactical_action <= 10000:
                df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\sequence\\" + competition_name + "\\others\\" + str(no_tactical_action).zfill(6) + ".csv")
                print(str(no_tactical_action).zfill(6))'''




if __name__ == "__main__":
    main()