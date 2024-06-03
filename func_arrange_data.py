# 360_dataの整形
def arrange_data(df, main_team_id):

    three_sixty_data = df.loc[:, '360_data'].copy()

    df.loc[:, ['teammate_count']] = 0
    df.loc[:, ['opponent_count']] = 0

    # 新しい列を作って -1.0 埋め
    for i in range(1,12):
        df.loc[:, ['teammate_' + str(i) + '_x']] = -1.0
        df.loc[:, ['teammate_' + str(i) + '_y']] = -1.0
        df.loc[:, ['opponent_' + str(i) + '_x']] = -1.0
        df.loc[:, ['opponent_' + str(i) + '_y']] = -1.0


    for i in range(len(three_sixty_data)):
        
        one_event_data = df.loc[i, '360_data']

        # one_event_data がfloatの時（中身が空の時）対策
        if isinstance(one_event_data, (list, tuple, str)):
            for j in range(len(one_event_data)):
                len_one_event_data = len(one_event_data)
        else:
            len_one_event_data = 0


        # which team does have possession
        # main team is attacking
        # コンバートしなくてよい
        if (df.loc[i, ['team_id']] == main_team_id).any().any():

            # チームメート一人目はボール位置
            df.loc[i, ['teammate_1_x']] = float(df.loc[i, ['start_x']].copy().iloc[0])
            df.loc[i, ['teammate_1_y']] = float(df.loc[i, ['start_y']].copy().iloc[0])

            # チームメート数
            teammate_count = 1
            # 相手チームメート数
            opponent_count = 0
            
            # 映っている選手いるか
            if three_sixty_data[i] == 0:
                continue
            else:

                for j in range(len_one_event_data):
                    one_person_data = one_event_data[j]

                    # チームメートかどうか
                    if one_person_data["teammate"] == True:
                        teammate_count += 1
                        df.loc[i, ['teammate_' + str(teammate_count) + '_x']] = one_person_data['location'][0]
                        df.loc[i, ['teammate_' + str(teammate_count) + '_y']] = one_person_data['location'][1]
                    else:
                        opponent_count += 1
                        df.loc[i, ['opponent_' + str(opponent_count) + '_x']] = one_person_data['location'][0]
                        df.loc[i, ['opponent_' + str(opponent_count) + '_y']] = one_person_data['location'][1]

                df.loc[i, 'teammate_count'] = teammate_count
                df.loc[i, 'opponent_count'] = opponent_count


        # main team doesn't have possession(opponent team has possession)
        # コンバートする
        else:

            # 相手チーム一人目はボール位置
            df.loc[i, ['opponent_1_x']] = 120.0 - float(df.loc[i, ['start_x']].copy().iloc[0])
            df.loc[i, ['opponent_1_y']] = float(df.loc[i, ['start_y']].copy().iloc[0])

            # チームメート数
            teammate_count = 0
            # 相手チームメート数
            opponent_count = 1
            
            # 映っている選手いるか
            if three_sixty_data[i] == 0:
                continue
            else:

                for j in range(len_one_event_data):
                    one_person_data = one_event_data[j]

                    # チームメートかどうか
                    if one_person_data["teammate"] == True:
                        opponent_count += 1
                        df.loc[i, ['opponent_' + str(opponent_count) + '_x']] = 120.0 - one_person_data['location'][0]
                        df.loc[i, ['opponent_' + str(opponent_count) + '_y']] = one_person_data['location'][1]
                    else:
                        teammate_count += 1
                        df.loc[i, ['teammate_' + str(teammate_count) + '_x']] = 120.0 - one_person_data['location'][0]
                        df.loc[i, ['teammate_' + str(teammate_count) + '_y']] = one_person_data['location'][1]

                df.loc[i, 'teammate_count'] = teammate_count
                df.loc[i, 'opponent_count'] = opponent_count