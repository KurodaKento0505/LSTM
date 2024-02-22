from func_possession import seg_possession

# shortcounter
def shortcounter(df_attack_team):

    # sequence_style_number
    a = 0

    # ボール奪取を定義
    if df_attack_team.iloc[[0],:].loc[:,["type_name"]].empty:
        a = 0
    # elif (df.iloc[[0],:].loc[:,["type_name"]].isin(["goalkick"]).any().any()) or (df.iloc[[0],:].loc[:,["type_name"]].isin(["throw_in"]).any().any()) or (df.iloc[[0],:].loc[:,["type_name"]].isin(["foul"]).any().any()) or (df.iloc[[0],:].loc[:,["type_name"]].isin(["corner_crossed"]).any().any()) or (df.iloc[[0],:].loc[:,["type_name"]].isin(["dribble"]).any().any()) or (df.iloc[[0],:].loc[:,["type_name"]].isin(["freekick_crossed"]).any().any()) or (df.iloc[[0],:].loc[:,["type_name"]].isin(["freekick_short"]).any().any()):
        # a = 0
    elif (df_attack_team.iloc[[0],:].loc[:,["type_name"]].isin(["interception"]).any().any()) or (df_attack_team.iloc[[0],:].loc[:,["type_name"]].isin(["tackle"]).any().any()):

        # ミドルサードもしくはアタッキングサード後方にボールが存在
        if (df_attack_team.iloc[[0],:].loc[:,["start_x"]] >= 35).any().any() and (df_attack_team.iloc[[0],:].loc[:,["start_x"]] <= 85).any().any():

            if (df_attack_team.iloc[[0],:].loc[:,["start_x"]] >= 70).any().any():
                for i in (n+1 for n in range(len(df_attack_team) - 1)):
                    if (df_attack_team.iloc[[i],:].loc[:,["start_x"]] > 85).any().any():
                        # print(df.iloc[[i],:].loc[:,["start_x"]])
                        # print(df.iloc[[i],:].loc[:,["start_x"]] > 70)

                        np_time_seconds = df_attack_team["time_seconds"].to_numpy().tolist()
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
                        a = 0
            
            else:
                # if time_secondsが10秒以内にアタッキングサードに行くか
                for i in (n+1 for n in range(len(df_attack_team) - 1)):
                    if (df_attack_team.iloc[[i],:].loc[:,["start_x"]] > 70).any().any():
                        # print(df.iloc[[i],:].loc[:,["start_x"]])
                        # print(df.iloc[[i],:].loc[:,["start_x"]] > 70)

                        np_time_seconds = df_attack_team["time_seconds"].to_numpy().tolist()
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
                        a = 0
            # 本当にインターセプションなどが起こった場所か確認必要
            # print(df['type_name'].iloc[0])
        else:
            a = 0

    return a



# セグメンテーション
# longcounter
def seg_shortcounter(df_attack_sequence, df_actions_half, start, end):
    
    # sequence_style_number
    a = 0

    # ボール奪取を定義
    if df_attack_sequence.loc[0,["type_name"]].empty:
        pass


    elif (df_attack_sequence.loc[0,["type_name"]].isin(["interception"]).any().any()) or (df_attack_sequence.loc[0,["type_name"]].isin(["tackle"]).any().any()):


        # ミドルサードもしくはアタッキングサード後方にボールが存在
        if (df_attack_sequence.loc[0,["start_x"]] >= 35).any().any() and (df_attack_sequence.loc[0,["start_x"]] <= 85).any().any():

            # 最初からアタッキングサードにボールある
            if (df_attack_sequence.loc[0,["start_x"]] >= 70).any().any():
                for k in range(len(df_attack_sequence)):
                    if (df_attack_sequence.loc[k,["start_x"]] > 85).any().any():

                        np_time_seconds = df_attack_sequence["time_seconds"].to_numpy().tolist()
                        end_timeseconds = np_time_seconds[k]
                        start_timeseconds = np_time_seconds[0]
                        time = end_timeseconds - start_timeseconds
                        
                        if time <= 10.0:
                            df_actions_half.loc[start : end, ['label']] = 2
                            a = 2

                            # shortcounterが終わったらpossessionかどうか調べる
                            seg_possession(df_attack_sequence, df_actions_half, start + k + 1, end)


            # ミドルサードにボール存在
            else:
                # if time_secondsが10秒以内にアタッキングサードに行くか
                for k in range(len(df_attack_sequence)):
                    if (df_attack_sequence.loc[k,["start_x"]] > 70).any().any():

                        np_time_seconds = df_attack_sequence["time_seconds"].to_numpy().tolist()
                        end_timeseconds = np_time_seconds[k]
                        start_timeseconds = np_time_seconds[0]
                        time = end_timeseconds - start_timeseconds
                        
                        if time <= 10.0:
                            df_actions_half.loc[start : end, ['label']] = 2
                            a = 2
                            
                            # shortcounterが終わったらpossessionかどうか調べる
                            seg_possession(df_attack_sequence, df_actions_half, start + k + 1, end)
    if a ==2:
        print('shortcounter')


    return a