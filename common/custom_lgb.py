import lightgbm as lgb
import custom_metrics

def fit(train_data, test_data):
    y_train = train_data['visitors']
    y_test = test_data['visitors']
    x_train = train_data.drop('visitors', axis=1)
    x_test = test_data.drop('visitors', axis=1)

    cat_col = ["air_store_num", "dow", "dowh", "dows", "air_genre_num", "air_area_num",
               "year", "month", "week", "day",
               "prefecture_num", "city_num",]

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {
        'learning_rate': 0.02,
        'num_leaves': 63,
#        'lambda_l1': 16.7,
        'boosting': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'feature_fraction': .3,
        'seed': 99,
        'verbose': 0
    }
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_eval,
                    verbose_eval=50,
                    early_stopping_rounds=200,
                    categorical_feature=cat_col)
    print('End training...')

    return gbm
