import numpy as np
import pandas as pd

import custom_metrics
import custom_lgb


# load data
train = pd.read_csv('../output/reg_train.csv')
predict = pd.read_csv('../output/reg_predict.csv')


# make input
train['visitors'] = np.log1p(train['visitors'])
train_input = train[ (train['visit_date'] >= '2016-01-01') & (train['visit_date'] < '2016-11-28') ].reset_index(drop=True)
test_input = train[ (train['visit_date'] >= '2017-01-16') & (train['visit_date'] < '2017-03-05') ].reset_index(drop=True)

col = ['air_store_num', 'visitors', 'air_genre_num', 'air_area_num',
       'year', 'month', 'min', 'max', 'median', 'mean', 'std',
       '3month_min', '3month_max', '3month_median', '3month_mean',
       '6month_min', '6month_max', '6month_median', '6month_mean',
       '12month_min', '12month_max', '12month_median', '12month_mean',
       'dow', 'dowh', 'holiday_flg', 'week_hols', 'next_week_hols', 'prev_week_hols',
       'quarter_regress', 'year_regress',
       "reserve_sum_air", "reserve_mean_air", "reserve_datediff_mean_air",
       "reserve_sum_hpg", "reserve_mean_hpg", "reserve_datediff_mean_hpg",
       "total_reserve_sum", 'total_reserve_mean', 'total_reserve_dt_diff_mean',
       ]
train_input = train_input[col]
test_input = test_input[col]

# print(train_input.head())
# print(test_input.head())
# print(x_pred.head())
# print('-' * 30)

# fit and predict
model = custom_lgb.fit(train_input, test_input)
x_pred = predict[col].drop('visitors', axis=1)
y_pred = model.predict(x_pred, num_iteration=model.best_iteration)
# y_pred[y_pred < 0] = 0
y_pred = np.expm1(y_pred)


# validate
def validate(validate_data, model):
    x_valid = validate_data[col].drop('visitors', axis=1)
    y_valid = model.predict(x_valid)
    validation_score = custom_metrics.rmse(validate_data["visitors"], y_valid)
    print(validation_score)


print("Validation score by six week:")
six_week_validate_data = train[train['visit_date'] >= '2017-03-05'].reset_index(drop=True)
validate(six_week_validate_data, model)

print("Validation score by last week:")
public_lb_validation_data = \
    train[train['visit_date'] >= '2017-04-16'].reset_index(drop=True)
validate(public_lb_validation_data, model)

fi = pd.DataFrame({"name": model.feature_name(), "importance": model.feature_importance()})
print(fi)

# submit
submission = pd.DataFrame({
        "id": predict['id'],
        "visitors": y_pred
    })
print(submission.describe())
submission.to_csv('../output/submission.csv',float_format='%.6f', index=False)

