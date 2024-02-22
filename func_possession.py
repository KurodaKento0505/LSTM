def seg_possession(df_attack_sequence, df_actions_half, start, end):
    
    # sequence_style_number
    a = 0

    # その他の攻撃のラベルを入れておく
    df_actions_half.loc[start : end, ['label']] = 5


    for i in range(len(df_attack_sequence)):


        # 敵陣にボールが存在
        if (df_attack_sequence.loc[i,["start_x"]] >= 52.5).any().any():

            # 時間を記憶
            possession_start_time = df_attack_sequence.loc[i,["time_seconds"]]

            # 20秒経つまで探す
            for j in range(i, len(df_attack_sequence)):

                # 敵陣から出たらお終い
                if (df_attack_sequence.loc[j,["start_x"]] < 52.5).any().any():
                    break

                # 20秒経った場合
                if (df_attack_sequence.loc[j,["time_seconds"]] - possession_start_time >= 20).any().any():

                    a = 3

                    # 20秒経った後，敵陣から出なかった場合
                    df_actions_half.loc[start + i : end, ['label']] = 3
                    
                    # 20秒経った後，敵陣から出た場合
                    for k in range(j, len(df_attack_sequence)):

                        # 敵陣から出たか否か
                        if (df_attack_sequence.loc[k,["start_x"]] < 52.5).any().any():

                            df_actions_half.loc[start + i : start + j + k, ['label']] = 3
                            df_actions_half.loc[start + j + k : end, ['label']] = 5
                            break


        # 自陣にボールが存在
        elif (df_attack_sequence.loc[i,["start_x"]] <= 52.5).any().any():

            # 時間を記憶
            possession_start_time = df_attack_sequence.loc[i,["time_seconds"]]

            # 20秒経つまで探す
            for j in range(i, len(df_attack_sequence)):

                # 自陣から出たらお終い
                if (df_attack_sequence.loc[j,["start_x"]] > 52.5).any().any():
                    break

                # 20秒経った場合
                if (df_attack_sequence.loc[j,["time_seconds"]] - possession_start_time >= 20).any().any():

                    a = 4

                    # 20秒経った後，自陣から出なかった場合
                    df_actions_half.loc[start + i : end, ['label']] = 4
                    
                    # 20秒経った後，自陣から出た場合
                    for k in range(j, len(df_attack_sequence)):

                        # 自陣から出たか否か
                        if (df_attack_sequence.loc[k,["start_x"]] > 52.5).any().any():

                            df_actions_half.loc[start + i : start + j + k, ['label']] = 4
                            df_actions_half.loc[start + j + k : end, ['label']] = 5
                            break


    if a == 3:
        print('opposition_half_possession')

    if a == 4:
        print('own_half_possession')



    return a