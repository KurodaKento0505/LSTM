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
                if df_in_play.loc[i,["start_x"]] <= 35:

                    # time_secondsが15秒以内にアタッキングサードに行くか
                    # ボール奪取した時刻
                    start_timeseconds = float(df_in_play.loc[i,["time_seconds"]])

                    for j in range(i + 1, len(df_in_play)):

                        # 相手ボールになったら終わり
                        if df_in_play.loc[j,["team_id"]] != main_team_id:
                            break

                        # アタッキングサードに侵入
                        if df_in_play.loc[j,["start_x"]] >= 70:

                            end_timeseconds = float(df_in_play.loc[j,["time_seconds"]])
                            time = end_timeseconds - start_timeseconds
                            print(time, end_timeseconds, start_timeseconds)
                            # np_time_seconds = df_in_play["time_seconds"].to_numpy().tolist()
                            # end_timeseconds = np_time_seconds[j]
                            # start_timeseconds = np_time_seconds[0]
                            # time = end_timeseconds - start_timeseconds
                            
                            if time <= 15.0:
                                df_in_play.loc[i : i + j, ['label']] = 1
                                a = 1

                            break

                '''# shortcounter
                # ミドルサードもしくはアタッキングサード後方にボールが存在
                elif (df_in_play.iloc[[0],:].loc[:,["start_x"]] >= 35).any().any() and (df_in_play.iloc[[0],:].loc[:,["start_x"]] <= 85).any().any():

                    if (df_in_play.iloc[[0],:].loc[:,["start_x"]] >= 70).any().any():
                        for i in (n+1 for n in range(len(df_in_play) - 1)):
                            if (df_in_play.iloc[[i],:].loc[:,["start_x"]] > 85).any().any():
                                # print(df.iloc[[i],:].loc[:,["start_x"]])
                                # print(df.iloc[[i],:].loc[:,["start_x"]] > 70)

                                np_time_seconds = df_in_play["time_seconds"].to_numpy().tolist()
                                end_timeseconds = np_time_seconds[i]
                                start_timeseconds = np_time_seconds[0]
                                time = end_timeseconds - start_timeseconds
                                
                                if time <= 10.0:
                                    # print(time)
                                    a = 2
                                    break

                                else:
                                    a = 0
                                

                            else:
                                a = 0'''


    # counterpress(label = 6)
    for i in range(len(df_in_play)):

        # main_teamかどうか
        if df_in_play.loc[i, ['team_id']] == main_team_id:

            # statsbomb のラベルを使用
            if df_in_play.loc[i, ['counterpress']] == True:
                df_in_play.loc[i, ['label']] = 6

    a = 0
    return a