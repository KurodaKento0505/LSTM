def label_attack_or_defense(df_in_play, main_team_id):

    for i in range(len(df_in_play)):

        if (df_in_play.loc[i, 'team_id'] == main_team_id).any().any():

            df_in_play.loc[i, 'attack'] = 1
        
        else:

            df_in_play.loc[i, 'defense'] = 1