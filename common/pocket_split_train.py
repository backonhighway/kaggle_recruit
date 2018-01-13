import custom_lgb


def split_train(splitted_df):
    models = []
    score = 0.0
    for split_df in splitted_df:
        train = split_df[0]
        test = split_df[1]
        model = custom_lgb.fit(train, test)
        score = score + model.best_score.get('valid_0').get('rmse')
        models.append(model)
    score = score / len(splitted_df)
    print("Average score:")
    print(score)

    return models


def split_set(df, period_list, col):
    splits = []
    for period in period_list:
        train_input = cut_input(df, period[0], period[1])
        test_input = cut_input(df,  period[2], period[3])
        train_input = train_input[col]
        test_input = test_input[col]
        splits.append([train_input, test_input])

    return splits


def cut_input(df, from_date, to_date):
    ret_df = df[(df['visit_date'] >= from_date) & (df['visit_date'] <= to_date)].reset_index(drop=True)
    return ret_df
