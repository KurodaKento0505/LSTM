import pandas as pd
from socceraction.data.statsbomb import StatsBombLoader
import math
import numpy as np

import socceraction.spadl as spadl

import matplotlib.pyplot as plt
import matplotsoccer as mps

from func_longcounter import seg_longcounter
from func_shortcounter import seg_shortcounter
from func_possession import seg_possession
from func_convert import convert_team
from func_convert import convert_left_to_right
from func_convert import convert_ball_left_to_right
from socceraction_dev_no_360 import time_reset
from socceraction_dev_no_360 import normalization
from socceraction_dev_360 import arrange_360_data

# Set up the StatsBomb data loader
SBL = StatsBombLoader()

# View all available competitions
df_competitions = SBL.competitions()

competition_id = 43
season_id = 106
competition_name = "FIFA_World_Cup_2022" # FIFA_World_Cup_2022, UEFA_Euro_2020, UEFA_Women's_Euro_2022, Women's_World_Cup_2023

# Create a dataframe with all games from UEFA Euro
df_games = SBL.games(competition_id, season_id).set_index("game_id")

#######################################################################################ここ変える########################################################
# 一つの大会におけるcount sequence
count_sequence = 0

# 一つの大会におけるlongcounter
include_longcounter = 0

# 一つの大会におけるshortcounter
include_shortcounter = 0

# 一つの大会におけるopposition_half_possession
include_opposition_half_possession = 0

# 一つの大会におけるown_half_possession
include_own_half_possession = 0

# 一つの大会におけるtactical_action
no_tactical_action = 10000

# in playの長さ
in_play_np = np.empty([0])

# longcounterの長さ
longcounter_length_np = np.empty([0])

# shortcounterの長さ
shortcounter_length_np = np.empty([0])

def main():

    global in_play_np
    global longcounter_length_np
    global shortcounter_length_np

    # convert するか否か
    convert = True


    for i in reversed(range(len(df_games.index))): # reversed

        '''if i == 0 or i == 1 or i == 2:
            continue'''
        
        '''if i == len(df_games.index) - 1: # or i == len(df_games.index) - 2 or i == len(df_games.index) - 3:
            continue'''

        game_id = df_games.index[i]
        df_teams = SBL.teams(game_id)
        df_players = SBL.players(game_id)
        df_events = SBL.events(game_id, True)  # 360dataが含まれる = True
        print(df_teams)

        home_team_id = df_games.at[game_id, "home_team_id"]


        # spdlによって、df_eventsをdf_actionsに変更
        df_actions = spadl.statsbomb.convert_to_actions(df_events, home_team_id)

        # Replace result, actiontype and bodypart IDs by their corresponding name
        df_actions = spadl.add_names(df_actions)

        # Add team and player names
        df_actions = df_actions.merge(df_teams).merge(df_players)
        
        # period_id,timestampで並べ替え
        df_actions = df_actions.sort_values(['period_id','time_seconds']) 


        # 前半 left to right するチームの id 
        convert_team_id_1st_half = convert_team(df_actions)

        # 後半 left to right するチームの id 
        if convert_team_id_1st_half == df_teams['team_id'].iloc[0]:
            convert_team_id_2nd_half = df_teams['team_id'].iloc[1]
            main_team_id_1st_half = df_teams['team_id'].iloc[1]
            main_team_id_2nd_half = df_teams['team_id'].iloc[0]
        else:
            convert_team_id_2nd_half = df_teams['team_id'].iloc[0]
            main_team_id_1st_half = df_teams['team_id'].iloc[0]
            main_team_id_2nd_half = df_teams['team_id'].iloc[1]

        print(main_team_id_1st_half,convert_team_id_1st_half,main_team_id_2nd_half,convert_team_id_2nd_half)


        # データを選別、'360_data'、'possession'、'player_name'
        df_actions = df_actions.loc[:,['period_id','time_seconds','team_id','start_x','start_y','end_x','end_y','type_name','play_pattern_name','result_name','team_name', '360_data']]
        
        # df_actions['attack_team_id'] = 0

        df_actions['label'] = 5

        df_actions['label_length'] = 0

        df_actions['until_longcounter'] = 1

        df_actions['until_shortcounter'] = 1

        df_actions['until_own_half_possession'] = 1

        df_actions['until_opposition_half_possession'] = 1

        df_actions['until_others'] = 1

        # df_actions['sequence_label'] = 0


        # 前後半で判別
        df_actions_1sthalf = df_actions[df_actions["period_id"] == 1]
        df_actions_2ndhalf = df_actions[df_actions["period_id"] == 2]

        # indexの振り直し
        df_actions_1sthalf.reset_index(drop=True, inplace=True)
        df_actions_2ndhalf.reset_index(drop=True, inplace=True)
        

        df_actions_1sthalf.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\test_data\\"+ competition_name +"\\df_1st_half.csv")
        df_actions_2ndhalf.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\data_42000\\test_data\\"+ competition_name +"\\df_2nd_half.csv")
        break

        if convert:
            arrange_360_data(df_actions_1sthalf, convert_team_id_1st_half)
            arrange_360_data(df_actions_2ndhalf, convert_team_id_2nd_half)
        else:
            arrange_360_data(df_actions_1sthalf, main_team_id_1st_half)
            arrange_360_data(df_actions_2ndhalf, main_team_id_2nd_half)

        # 前半
        label(df_actions_1sthalf, main_team_id_1st_half, convert_team_id_1st_half, convert)

        # 後半
        label(df_actions_2ndhalf, main_team_id_2nd_half, convert_team_id_2nd_half, convert)
        
        # counter_length(df_actions_1sthalf)
        # ounter_length(df_actions_2ndhalf)

        # 前半
        sequence_spilit(df_actions_1sthalf, main_team_id_1st_half, convert_team_id_1st_half, convert)

        # 後半
        sequence_spilit(df_actions_2ndhalf, main_team_id_2nd_half, convert_team_id_2nd_half, convert)


        print((i + 1) / len(df_games))
        break


    # 一つの大会におけるin playの長さ、longcounter、shortcounterの長さ
    #csvファイルとして保存
    # np.savetxt('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\data\\' + competition_name + '\\longcounter_length_np.csv', longcounter_length_np, delimiter=',')
    # np.savetxt('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\data\\' + competition_name + '\\shortcounter_length_np.csv', shortcounter_length_np, delimiter=',')
    # np.savetxt('C:\\Users\\黒田堅仁\\OneDrive\\My_Research\\Dataset\\StatsBomb\\segmentation\\data\\' + competition_name + '\\in_play_np.csv', in_play_np, delimiter=',')





# 一つの大会でのlongcounterの個数
b = 0

# 一つの大会でのshortcounterの個数
c = 0

# 一つの大会でのno_counterの個数
d = 0

# 局面ラベル
sequence_style_number = 0


# 攻撃シーケンス分類かつそれぞれの局面関数に移動させてラベル付与
def label(df_actions_half, main_team_id, convert_team_id, convert):

    global b # 大会でのlongcounterの個数
    global c # 大会でのlongcounterの個数
    global d # 大会でのno_counterの個数


    next_in_play_start = 0

    for j in range(len(df_actions_half) - 1):


        # １．out of playを区切る

        # NaNの対処
        if (df_actions_half.loc[j, ['play_pattern_name']].isnull() == True).any().any():
            pass
        elif (df_actions_half.loc[j + 1, ['play_pattern_name']].isnull() == True).any().any():
            pass

        # play_pattern_nameが変わったらout of play
        elif (df_actions_half.loc[j, ['play_pattern_name']] != df_actions_half.loc[j + 1, ['play_pattern_name']]).any().any() or (df_actions_half.loc[j + 1, ['type_name']] == 'throw_in').any().any() or (df_actions_half.loc[j + 1, ['type_name']] == 'corner_kick').any().any() or (df_actions_half.loc[j + 1, ['type_name']] == 'goalkick').any().any() or (df_actions_half.loc[j + 1, ['type_name']] == 'freekick_crossed').any().any():

            # 次の play_pattern_name が regular play の場合、継続
            if (df_actions_half.loc[j + 1, ['play_pattern_name']].isin(['Regular Play']) == True).any().any():
                continue

            # 次の play_pattern_name が From Counter の場合、継続
            if (df_actions_half.loc[j + 1, ['play_pattern_name']].isin(['From Counter']) == True).any().any():
                continue

            df_in_play = df_actions_half[next_in_play_start : j + 1]

            # indexの振り直し
            df_in_play.reset_index(drop=True, inplace=True)

            next_attack_sequence_start = 0


            # ２．攻撃シーケンスを区切る
            for k in range(len(df_in_play) - 1):

                # 攻撃チームidを記憶
                attack_team_id = df_in_play.loc[next_attack_sequence_start,["team_id"]]


                # in playが終わったら、最後までを一つのシーケンスに
                if k == len(df_in_play) - 2:
                    df_attack_sequence = df_in_play[next_attack_sequence_start : k + 2]

                    # indexの振り直し
                    df_attack_sequence.reset_index(drop=True, inplace=True)
                    
                    # df_actions_halfにおける始まりと終わり
                    start = next_in_play_start + next_attack_sequence_start
                    end = next_in_play_start + k + 1

                    # convert
                    # convert_team_id じゃなかったら守備
                    if convert:
                        if (attack_team_id == convert_team_id).any().any():
                            convert_ball_left_to_right(df_attack_sequence)

                            # longcounter かどうか
                            sequence_style_number = seg_longcounter(df_attack_sequence, df_actions_half, start, end)

                            # shortcounter かどうか
                            if sequence_style_number != 1:
                                sequence_style_number = seg_shortcounter(df_attack_sequence, df_actions_half, start, end)

                                # opposition_half_possession, own_half_possession かどうか
                                if sequence_style_number != 2:
                                    seg_possession(df_attack_sequence, df_actions_half, start, end)

                            # convert戻す
                            if (attack_team_id == convert_team_id).any().any():
                                convert_ball_left_to_right(df_attack_sequence)

                        else:
                            df_actions_half.loc[start : end, ['label']] = 0

                    # main_team_id じゃなかったら守備
                    else:
                        if (attack_team_id == main_team_id).any().any():

                            # longcounter かどうか
                            sequence_style_number = seg_longcounter(df_attack_sequence, df_actions_half, start, end)

                            # shortcounter かどうか
                            if sequence_style_number != 1:
                                sequence_style_number = seg_shortcounter(df_attack_sequence, df_actions_half, start, end)

                                # opposition_half_possession, own_half_possession かどうか
                                if sequence_style_number != 2:
                                    seg_possession(df_attack_sequence, df_actions_half, start, end)

                        else:
                            df_actions_half.loc[start : end, ['label']] = 0


                    # attack_team_idを保存
                    for l in range(start,end + 1):
                        df_actions_half.loc[l, ['attack_team_id']] = float(attack_team_id)


                # 二回連続で違うチームが来たら相手チームの攻撃シーケンス
                elif (df_in_play.loc[k,["team_id"]] != attack_team_id).any().any():

                    if (df_in_play.loc[k + 1,["team_id"]] != attack_team_id).any().any():

                        df_attack_sequence = df_in_play[next_attack_sequence_start : k]

                        # indexの振り直し
                        df_attack_sequence.reset_index(drop=True, inplace=True)

                        # df_actions_halfにおける始まりと終わり
                        start = next_in_play_start + next_attack_sequence_start
                        end = next_in_play_start + k - 1

                        # convert
                        # convert_team_id じゃなかったら守備
                        if convert:
                            if (attack_team_id == convert_team_id).any().any():
                                convert_ball_left_to_right(df_attack_sequence)

                                # longcounter かどうか
                                sequence_style_number = seg_longcounter(df_attack_sequence, df_actions_half, start, end)

                                # shortcounter かどうか
                                if sequence_style_number != 1:
                                    sequence_style_number = seg_shortcounter(df_attack_sequence, df_actions_half, start, end)

                                # convert戻す
                                if (attack_team_id == convert_team_id).any().any():
                                    convert_ball_left_to_right(df_attack_sequence)

                            else:
                                df_actions_half.loc[start : end, ['label']] = 0

                        # main_team_id じゃなかったら守備
                        else:
                            if (attack_team_id == main_team_id).any().any():

                                # longcounter かどうか
                                sequence_style_number = seg_longcounter(df_attack_sequence, df_actions_half, start, end)

                                # shortcounter かどうか
                                if sequence_style_number != 1:
                                    sequence_style_number = seg_shortcounter(df_attack_sequence, df_actions_half, start, end)

                            else:
                                df_actions_half.loc[start : end, ['label']] = 0

                        # attack_team_idを保存
                        for l in range(start,end + 1):
                            df_actions_half.loc[l, ['attack_team_id']] = float(attack_team_id)

                        next_attack_sequence_start = k

            next_in_play_start = j + 1



# 注目データの前後5データを一つのシーケンスに
def sequence_spilit(df_actions_half, main_team_id, convert_team_id, convert):

    global in_play_np

    # 一つの大会におけるcount sequence
    global count_sequence

    # 一つの大会におけるno_counter、include_long shortcounter
    global include_longcounter
    global include_shortcounter
    global include_opposition_half_possession
    global include_own_half_possession
    global no_tactical_action

    # data_length
    data_length = 10

    next_in_play_start = 0

    for i in range(len(df_actions_half) - 1):

        # out of playを区切る

        # NaNの対処
        if (df_actions_half.loc[i, ['play_pattern_name']].isnull() == True).any().any():
            pass
        elif (df_actions_half.loc[i + 1, ['play_pattern_name']].isnull() == True).any().any():
            pass
        elif (df_actions_half.loc[i, ['play_pattern_name']] != df_actions_half.loc[i + 1, ['play_pattern_name']]).any().any() or (df_actions_half.loc[i + 1, ['type_name']] == 'throw_in').any().any() or (df_actions_half.loc[i + 1, ['type_name']] == 'corner_kick').any().any() or (df_actions_half.loc[i + 1, ['type_name']] == 'goalkick').any().any() or (df_actions_half.loc[i + 1, ['type_name']] == 'freekick_crossed').any().any():

            # 次の play_pattern_name が regular play の場合、継続
            if (df_actions_half.loc[i + 1, ['play_pattern_name']].isin(['Regular Play']) == True).any().any():
                continue

            # 次の play_pattern_name が From Counter の場合、継続
            if (df_actions_half.loc[i + 1, ['play_pattern_name']].isin(['From Counter']) == True).any().any():
                continue

            df_in_play = df_actions_half[next_in_play_start : i + 1]

            # indexの振り直し
            df_in_play.reset_index(drop=True, inplace=True)

            # in playの長さ
            # in_play_np = np.append(in_play_np, len(df_in_play))

            
            if len(df_in_play) <= data_length:
                next_in_play_start += len(df_in_play)
                continue


            # 注目データと前5データと後5データの合計11データを一つのシーケンスに
            for j in range(int(data_length/2), len(df_in_play) - int(data_length/2)):

                # sequence分割
                df_sequence = df_in_play[ j - int(data_length / 2) : j + int(data_length / 2) + 1 ].copy()

                # indexの振り直し
                df_sequence.reset_index(drop=True, inplace=True)

                '''# others が 10000 を超えているときに実行
                # 戦術的行動が含まれるかどうか
                for k in range(data_length + 1):
                    if (df_sequence.loc[k, ['label']] == 5).any().any():
                        include_tactical_action = False
                    else:
                        include_tactical_action = True
                        break

                # 戦術的行動が含まれない（もう10000超えているためスキップ）
                if include_tactical_action != True:
                    continue'''


                # sequence の中から各戦術的行動を見つける

                # longcounter と shortcounter のみ to_csv するための変数
                sequence_type = 0

                # longcounter
                for k in range(data_length + 1):

                    if (df_sequence.loc[k, ['label']] == 1).any().any():

                        df_sequence.loc[0, ['until_longcounter']] = (k - 5) / 5

                        include_tactical_action = True

                        sequence_type = 1 # include_longcounter

                        break

                # shortcounter
                for k in range(data_length + 1):
                    
                    if (df_sequence.loc[k, ['label']] == 2).any().any():

                        df_sequence.loc[0, ['until_shortcounter']] = (k - 5) / 5

                        include_tactical_action = True

                        sequence_type = 2 # include_shortcounter

                        break

                # opposition_half_possession
                for k in range(data_length + 1):

                    if (df_sequence.loc[k, ['label']] == 3).any().any():

                        df_sequence.loc[0, ['until_opposition_half_possession']] = (k - 5) / 5

                        include_tactical_action = True

                        break

                # own_half_possession
                for k in range(data_length + 1):

                    if (df_sequence.loc[k, ['label']] == 4).any().any():

                        df_sequence.loc[0, ['until_own_half_possession']] = (k - 5) / 5

                        include_tactical_action = True

                        break
                
                # own_others
                for k in range(data_length + 1):

                    if (df_sequence.loc[k, ['label']] == 5).any().any():

                        df_sequence.loc[0, ['until_others']] = (k - 5) / 5

                        break
                        
                # convert
                if convert:
                    convert_left_to_right(df_sequence)
                
                # seg_to_csv
                seg_to_csv(df_sequence, convert, sequence_type) # test の場合 count_sequence


            next_in_play_start = i + 1


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



# それぞれのシーケンスをto_csv
def seg_to_csv(df_sequence, convert, sequence_type):

    global competition_name
    global count_sequence
    global include_longcounter
    global include_shortcounter
    global include_opposition_half_possession
    global include_own_half_possession
    global no_tactical_action

    df_sequence = df_sequence.loc[:,['start_x', 'start_y', 'time_seconds', 'label', 'until_longcounter', 'until_shortcounter', 'until_opposition_half_possession', 'until_own_half_possession', 'until_others', 'teammate_count', 'opponent_count', 'teammate_1_x', 'teammate_1_y', 'opponent_1_x', 'opponent_1_y', 'teammate_2_x', 'teammate_2_y', 'opponent_2_x', 'opponent_2_y', 'teammate_3_x', 'teammate_3_y', 'opponent_3_x', 'opponent_3_y', 'teammate_4_x', 'teammate_4_y', 'opponent_4_x', 'opponent_4_y', 'teammate_5_x', 'teammate_5_y', 'opponent_5_x', 'opponent_5_y', 'teammate_6_x', 'teammate_6_y', 'opponent_6_x', 'opponent_6_y', 'teammate_7_x', 'teammate_7_y', 'opponent_7_x', 'opponent_7_y', 'teammate_8_x', 'teammate_8_y', 'opponent_8_x', 'opponent_8_y', 'teammate_9_x', 'teammate_9_y', 'opponent_9_x', 'opponent_9_y', 'teammate_10_x', 'teammate_10_y', 'opponent_10_x', 'opponent_10_y', 'teammate_11_x', 'teammate_11_y', 'opponent_11_x', 'opponent_11_y']] # time_seconds, 'until_possession'
    df_sequence = time_reset(df_sequence)
    df_sequence = normalization(df_sequence)

    if convert:
        count_sequence += 1
        df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\test_sequence\\" + competition_name + "_convert\\" + str(count_sequence).zfill(6) + ".csv")
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
            print(str(include_shortcounter).zfill(6))'''

        '''elif (df_sequence.loc[0, ['until_opposition_half_possession']] != 1).any().any():
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
        df_sequence.to_csv("C:\\Users\\黒田堅仁\\OneDrive\\My_Research\Dataset\\StatsBomb\\segmentation\\add_player\\when_start_point\\counter_possession_others\\test_sequence\\" + competition_name + "\\" + str(count_sequence).zfill(6) + ".csv")
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
            print(str(include_shortcounter).zfill(6))'''

        '''elif (df_sequence.loc[0, ['until_opposition_half_possession']] != 1).any().any():
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