def label_time_by_starting(df_in_play, main_team_id, tactical_action_name_list):
    
    # 戦術的行動の開始時刻の探索
    for i in range(len(tactical_action_name_list)):

        next_j = 0
        
        for j in range(len(df_in_play)):

            # next_j から開始する場合
            if j < next_j:
                continue

            # 開始時刻を発見
            if (df_in_play.loc[j, tactical_action_name_list[i]] == 1).any().any():

                # 時間を記憶
                start_time = float(df_in_play.loc[j, 'time_seconds'])

                # インプレーの最初から戦術的行動の終了 or インプレー終了までラベル付け
                for k in range(len(df_in_play)):

                    # 終了条件
                    if (k > j) and (df_in_play.loc[k, tactical_action_name_list[i]] == 0).any().any():
                        next_j = k
                        break

                    # 開始時刻までの時間
                    time_to_start = float(df_in_play.loc[k, 'time_seconds']) - start_time

                    # 開始時刻までの時間を挿入
                    # -20 より小さい場合は -20
                    if time_to_start < -20:
                        pass
                    else:
                        df_in_play.loc[k, 'time_to_' + tactical_action_name_list[i]] = int(time_to_start)

                    next_j = k