import lightgbm as lgb


def do_predict(train_data, test_data, x_pred):
    y_train = train_data['visitors']
    y_test = test_data['visitors']
    x_train = train_data.drop('visitors', axis=1)
    x_test = test_data.drop('visitors', axis=1)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {
        'learning_rate': 0.05,
        'num_leaves': 31,
#        'lambda_l1': 16.7,
        'boosting': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'feature_fraction': .7,
        'seed': 99,
        'verbose': 0
    }
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,
                    valid_sets=lgb_eval,
                    verbose_eval=50,
                    early_stopping_rounds=100)

    # predict
    print('Start predicting...')
    y_pred = gbm.predict(x_pred, num_iteration=gbm.best_iteration)

    return y_pred
