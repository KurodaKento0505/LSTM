# 攻撃を全て左から右に変換するチーム名を知る関数
def convert_team(df):

    # keeper の位置によって変換するチームを決める
    three_sixty_data = df['360_data']

    for i in range(len(three_sixty_data)):
        one_event_data = three_sixty_data[i]

        for j in range(len(one_event_data)):
            one_person_data = one_event_data[j]

            if one_person_data["keeper"] == True:

                # チームメートかどうか
                if one_person_data["teammate"] == True:
                    print(one_person_data['location'][0], one_person_data['location'][1])
                    convert_team_id = df.loc[i, ['team_id']]

    '''if df['end_x'].iloc[0] >= 52.5:
        convert_team_id = df['team_id'].iloc[0]
    else:
        i = 0
        for i in range(len(df)):
            if i == 0:
                continue
            elif df['team_id'].iloc[i] != df['team_id'].iloc[i - 1]:
                convert_team_id = df['team_id'].iloc[i - 1]
            else:
                continue'''
    
    return convert_team_id


# 攻撃を全て左から右に変換するチームを変換する関数
# ボールと選手どっちも
def convert_left_to_right(df):# a = length

    for i in range(len(df)):

        # convert ball position
        before_start_x = df.at[i,'start_x']
        df.at[i,'start_x'] = 105.0 - before_start_x

        before_end_x = df.at[i,'end_x']
        df.at[i,'end_x'] = 105.0 - before_end_x

        # convert player position
        teammate_x_list = ['teammate_1_x','teammate_2_x','teammate_3_x','teammate_4_x','teammate_5_x','teammate_6_x','teammate_7_x','teammate_8_x','teammate_9_x','teammate_10_x','teammate_11_x']
        opponent_player_x_list = ['opponent_1_x','opponent_2_x','opponent_3_x','opponent_4_x','opponent_5_x','opponent_6_x','opponent_7_x','opponent_8_x','opponent_9_x','opponent_10_x','opponent_11_x']
        before_teammate_x = [0] * 12
        before_opponent_player_x = [0] * 12
        
        for j in range(1,12):
            if df.at[i,teammate_x_list[j - 1]] == -1.0:
                continue
            else:
                before_teammate_x[j] = df.at[i,teammate_x_list[j - 1]]
                df.at[i,teammate_x_list[j - 1]] = 105.0 - before_teammate_x[j]
    
            if df.at[i,opponent_player_x_list[j - 1]] == -1.0:
                continue
            else:
                before_opponent_player_x[j] = df.at[i,opponent_player_x_list[j - 1]]
                df.at[i,opponent_player_x_list[j - 1]] = 105.0 - before_opponent_player_x[j]


# ボールの位置のみ変換
def convert_ball_left_to_right(df):

    for i in range(len(df)):

        # convert ball position
        before_start_x = df.at[i,'start_x']
        df.at[i,'start_x'] = 105.0 - before_start_x

        before_end_x = df.at[i,'end_x']
        df.at[i,'end_x'] = 105.0 - before_end_x