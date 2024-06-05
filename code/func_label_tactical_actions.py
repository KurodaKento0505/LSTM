def label_tactical_actions(df_in_play, main_team_id):


    ################### longcounter or shortcounter ###################

    # skip して移るループ変数
    next_i = 0

    for i in range(1, len(df_in_play)):

        # skip
        if i < next_i:
            continue

        # ボール奪取
        # i-1番目は相手ボール and クリアじゃない
        if (df_in_play.loc[i - 1,["team_id"]] != main_team_id).any().any() and (df_in_play.loc[i - 1,["type_name"]] != 'clearance').any().any():
            # i番目は味方ボール
            if (df_in_play.loc[i,["team_id"]] == main_team_id).any().any():

                # ボール奪取した時刻
                start_timeseconds = float(df_in_play.loc[i,["time_seconds"]].iloc[0])


                ################### longcounter ###################
                # ボール位置がディフェンシブサードにボールが存在
                if (df_in_play.loc[i,["start_x"]] <= 40).any().any():

                    # time_secondsが15秒以内にアタッキングサードに行くか
                    # ボール奪取した時刻
                    start_timeseconds = float(df_in_play.loc[i,["time_seconds"]].iloc[0])

                    # 相手ボールになるか，アタッキングサードに侵入するまでループ
                    for j in range(i + 1, len(df_in_play)):

                        # 相手ボールになったら終わり
                        if (df_in_play.loc[j,["team_id"]] != main_team_id).any().any():

                            # skip
                            next_i = j + 1
                            break

                        # アタッキングサードに侵入
                        if (df_in_play.loc[j,["start_x"]] >= 80).any().any():

                            # 侵入した時刻
                            end_timeseconds = float(df_in_play.loc[j,["time_seconds"]].iloc[0])

                            # 侵入までに要した時間
                            time = end_timeseconds - start_timeseconds
                            
                            if time <= 15.0:
                                df_in_play.loc[i : j, ['longcounter']] = 1

                                # 続く限りラベル付け
                                for k in range(j + 1, len(df_in_play)):

                                    # 相手チームになる or アタッキングサードから出る で終わり
                                    if (df_in_play.loc[k,["team_id"]] != main_team_id).any().any() or (df_in_play.loc[k,["start_x"]] < 80).any().any():
                                        
                                        # 最初の行の場合は break
                                        if k == j + 1:
                                            break
                                        # 最初の行以外
                                        else:
                                            df_in_play.loc[j + 1 : k - 1, ['longcounter']] = 1
                                            break

                                    # 自分チーム and アタッキングサードのまま
                                    else:
                                        # 最後の行
                                        if k == len(df_in_play) - 1:
                                            df_in_play.loc[j + 1 : k, ['longcounter']] = 1
                                            
                                        # 最後の行以外
                                        else:
                                            continue

                            # skip
                            next_i = j + 1
                            break

                        # 最後
                        if j == len(df_in_play) - 1:
                            # skip
                            next_i = j + 1


                ################### shortcounter ###################
                # アタッキングサード後方にボールが存在
                elif (df_in_play.loc[i,["start_x"]] >= 80).any().any() and (df_in_play.loc[i,["start_x"]] <= 100).any().any():
                        
                    # time_secondsが10秒以内にアタッキングサード前方に行くか
                    # 相手ボールになるか，アタッキングサード前方に侵入するまでループ
                    for j in range(i + 1, len(df_in_play)):

                        # 相手ボールになったら終わり
                        if (df_in_play.loc[j,["team_id"]] != main_team_id).any().any():
                            
                            # skip
                            next_i = j + 1
                            break

                        # アタッキングサード前方に侵入
                        if (df_in_play.loc[j,["start_x"]] >= 100).any().any():

                            # 侵入した時刻
                            end_timeseconds = float(df_in_play.loc[j,["time_seconds"]].iloc[0])

                            # 侵入までに要した時間
                            time = end_timeseconds - start_timeseconds
                            
                            if time <= 10.0:
                                df_in_play.loc[i : j, ['shortcounter']] = 1

                                # 続く限りラベル付け
                                for k in range(j + 1, len(df_in_play)):

                                    # 相手チームになる or アタッキングサードから出る で終わり
                                    if (df_in_play.loc[k,["team_id"]] != main_team_id).any().any() or (df_in_play.loc[k,["start_x"]] < 80).any().any():
                                        
                                        # 最初の行の場合は break
                                        if k == j + 1:
                                            break
                                        # 最初の行以外
                                        else:
                                            df_in_play.loc[j + 1 : k - 1, ['shortcounter']] = 1
                                            break

                                    # 自分チーム and アタッキングサードのまま
                                    else:
                                        # 最後の行
                                        if k == len(df_in_play) - 1:
                                            df_in_play.loc[j + 1 : k, ['shortcounter']] = 1
                                            
                                        # 最後の行以外
                                        else:
                                            continue
                            
                            # skip
                            next_i = j + 1
                            break

                        # 最後
                        if j == len(df_in_play) - 1:
                            # skip
                            next_i = j + 1


                # ミドルサードにボールが存在
                elif (df_in_play.loc[i,["start_x"]] >= 40).any().any() and (df_in_play.loc[i,["start_x"]] <= 80).any().any():

                    # time_secondsが10秒以内にアタッキングサードに行くか
                    # 相手ボールになるか，アタッキングサードに侵入するまでループ
                    for j in range(i + 1, len(df_in_play)):

                        # 相手ボールになったら終わり
                        if (df_in_play.loc[j,["team_id"]] != main_team_id).any().any():
                            
                            # skip
                            next_i = j + 1
                            break

                        # アタッキングサードに侵入
                        if (df_in_play.loc[j,["start_x"]] >= 80).any().any():

                            # 侵入した時刻
                            end_timeseconds = float(df_in_play.loc[j,["time_seconds"]].iloc[0])

                            # 侵入までに要した時間
                            time = end_timeseconds - start_timeseconds
                            
                            if time <= 10.0:
                                df_in_play.loc[i : j, ['shortcounter']] = 1

                                # 続く限りラベル付け
                                for k in range(j + 1, len(df_in_play)):

                                    # 相手チームになる or アタッキングサードから出る で終わり
                                    if (df_in_play.loc[k,["team_id"]] != main_team_id).any().any() or (df_in_play.loc[k,["start_x"]] < 80).any().any():
                                        
                                        # 最初の行の場合は break
                                        if k == j + 1:
                                            break
                                        # 最初の行以外
                                        else:
                                            df_in_play.loc[j + 1 : k - 1, ['shortcounter']] = 1
                                            break

                                    # 自分チーム and アタッキングサードのまま
                                    else:
                                        # 最後の行
                                        if k == len(df_in_play) - 1:
                                            df_in_play.loc[j + 1 : k, ['shortcounter']] = 1
                                            
                                        # 最後の行以外
                                        else:
                                            continue
                            
                            # skip
                            next_i = j + 1
                            break

                        # 最後
                        if j == len(df_in_play) - 1:
                            # skip
                            next_i = j + 1


    ################### opposition_half_possession ###################

    # skip して移るループ変数
    next_i = 0

    for i in range(len(df_in_play)):

        # skip
        if i < next_i:
            continue

        # 味方ボール かつ 敵陣にボールが存在
        if (df_in_play.loc[i,["team_id"]] == main_team_id).any().any() and (df_in_play.loc[i,["start_x"]] >= 60).any().any():

            # 敵陣ポゼッション開始した時刻
            start_timeseconds = float(df_in_play.loc[i,["time_seconds"]].iloc[0])

            # 終了するまで探す
            for j in range(i + 1, len(df_in_play)):

                # 相手ボールになったら or 敵陣から出たら or 最後の行になったら 終わり
                if (df_in_play.loc[j,["team_id"]] != main_team_id).any().any() or (df_in_play.loc[j,["start_x"]] < 60).any().any() or (j == len(df_in_play) - 1):
                    
                    # 終了した時刻
                    end_timeseconds = float(df_in_play.loc[j,["time_seconds"]].iloc[0])

                    # 終了までに要した時間
                    time = end_timeseconds - start_timeseconds

                    # 20秒以上経ったら敵陣ポゼッション
                    if time >= 20.0:
                        df_in_play.loc[i : j, ['opposition_half_possession']] = 1
                        
                        # skip
                        next_i = j + 1
                        break

                    else:
                        # skip
                        next_i = j + 1
                        break



    ################### own_half_possession ###################

    # skip して移るループ変数
    next_i = 0

    for i in range(len(df_in_play)):

        # skip
        if i < next_i:
            continue

        # 味方ボール かつ 自陣にボールが存在
        if (df_in_play.loc[i,["team_id"]] == main_team_id).any().any() and (df_in_play.loc[i,["start_x"]] < 60).any().any():

            # 自陣ポゼッション開始した時刻
            start_timeseconds = float(df_in_play.loc[i,["time_seconds"]].iloc[0])

            # 終了するまで探す
            for j in range(i + 1, len(df_in_play)):

                # 相手ボールになったら or 自陣から出たら or 最後の行になったら終わり
                if (df_in_play.loc[j,["team_id"]] != main_team_id).any().any() or (df_in_play.loc[j,["start_x"]] >= 60).any().any() or (j == len(df_in_play) - 1):
                    
                    # 終了した時刻
                    end_timeseconds = float(df_in_play.loc[j,["time_seconds"]].iloc[0])

                    # 終了までに要した時間
                    time = end_timeseconds - start_timeseconds

                    # 20秒以上経ったら自陣ポゼッション
                    if time >= 20.0:
                        df_in_play.loc[i : j, ['own_half_possession']] = 1
                        
                        # skip
                        next_i = j + 1
                        break

                    else:
                        # skip
                        next_i = j + 1
                        break


    ################### counterpress ###################
    for i in range(len(df_in_play)):

        # main_team and counterpress発動 かどうか
        if (df_in_play.loc[i, ['team_id']] == main_team_id).any().any() and (df_in_play.loc[i, ['counterpress']] == True).any().any():
            
            df_in_play.loc[i, ['counterpressing']] = 1


    ################### highpressing or middlepressing ###################

    # skip して移るループ変数
    next_i = 0

    for i in range(len(df_in_play) - 1):

        # skip
        if i < next_i:
            continue

        # i 番目において，相手チームがボール保持 and pressを受けているかどうか
        if (df_in_play.loc[i, ['team_id']] != main_team_id).any().any() and (df_in_play.loc[i, ['under_pressure']] == True).any().any():

            # i + 1 番目において，相手チームがボール保持 and pressを受けているかどうか
            if (df_in_play.loc[i + 1, ['team_id']] != main_team_id).any().any() and (df_in_play.loc[i + 1, ['under_pressure']] == True).any().any():

                ################### highpressing ###################
                # pressを受けている選手がディフェンダーかどうか
                if float(df_in_play.loc[i, ['position_id']].iloc[0]) <= 8:

                    # 終了するまでループ
                    for j in range(i + 2, len(df_in_play)):

                        # 相手チームがボール保持 and pressを受けているかどうか
                        if (df_in_play.loc[j, ['team_id']] != main_team_id).any().any() and (df_in_play.loc[j, ['under_pressure']] == True).any().any():
                            
                            # 最後までプレス
                            if j == len(df_in_play) - 1:
                                df_in_play.loc[i : j, ['highpressing']] = 1
                                
                                # skip
                                next_i = j + 1
                            else:
                                continue

                        # 終了
                        else:
                            df_in_play.loc[i : j - 1, ['highpressing']] = 1

                            # skip
                            next_i = j + 1
                            break


                ################### middlepressing ###################
                # pressを受けている選手がディフェンダー以外かどうか
                else:

                    # 終了するまでループ
                    for j in range(i + 2, len(df_in_play)):

                        # 相手チームがボール保持 and pressを受けているかどうか
                        if (df_in_play.loc[j, ['team_id']] != main_team_id).any().any() and (df_in_play.loc[j, ['under_pressure']] == True).any().any():
                            
                            # 最後までプレス
                            if j == len(df_in_play) - 1:
                                df_in_play.loc[i : j, ['middlepressing']] = 1

                                # skip
                                next_i = j + 1
                            else:
                                continue

                        # 終了
                        else:
                            df_in_play.loc[i : j - 1, ['middlepressing']] = 1

                            # skip
                            next_i = j + 1
                            break