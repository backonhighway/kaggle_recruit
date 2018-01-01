import custom_lgb


def split_train(df, period_list):
    models = []
    score = 0.0
    for period in period_list:
        model = split_fit(df, period[0], period[1], period[2], period[3])
        score = score + model.best_score.get('valid_0').get('rmse')
        models.append(model)
    score = score / len(period_list)
    print("Average score= " + str(score))

    return models, score


def split_fit(df, train_from, train_to, test_from, test_to):
    train_input = df[
        (df['visit_date_str'] >= train_from) & (df['visit_date_str'] < train_to)].reset_index(drop=True)
    test_input = df[
        (df['visit_date_str'] >= test_from) & (df['visit_date_str'] < test_to)].reset_index(drop=True)

    col = ['air_store_num', 'visitors', 'dow', 'holiday_flg', 'air_genre_num', 'air_area_num']
    train_input = train_input[col]
    test_input = test_input[col]

    model = custom_lgb.fit(train_input, test_input)
    return model
