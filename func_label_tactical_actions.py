def label_tactical_actions(df_in_play, main_team_id):
    
    # longcounter(label = 1) or shortcounter(label= 2)
    for i in range(1, len(df_in_play)):

        # ボール奪取
        # i-1番目は相手ボール
        if df_in_play.loc[i - 1,["team_id"]] != main_team_id:
            # i番目は味方ボール
            if df_in_play.loc[i,["team_id"]] == main_team_id:

                # longcounter
                # ボール位置がディフェンシブサードにボールが存在
                if df_in_play.loc[0,["start_x"]] <= 35:

                    # if time_secondsが15秒以内にアタッキングサードに行くか
                    for j in range(len(df_in_play)):
                        if (df_in_play.loc[j,["start_x"]] >= 70).any().any():

                            np_time_seconds = df_in_play["time_seconds"].to_numpy().tolist()
                            end_timeseconds = np_time_seconds[j]
                            start_timeseconds = np_time_seconds[0]
                            time = end_timeseconds - start_timeseconds
                            
                            if time <= 15.0:
                                df_in_play.loc[i : i + j, ['label']] = 1
                                a = 1

                # shortcounter


    # counterpress(label = 6)
    for i in range(len(df_in_play)):

        # main_teamかどうか
        if df_in_play.loc[i, ['team_id']] == main_team_id:

            # statsbomb のラベルを使用
            if df_in_play.loc[i, ['counterpress']] == True:
                df_in_play.loc[i, ['label']] = 6

    a = 0
    return a