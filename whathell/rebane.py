import numpy as np
import pandas as pd

import custom_metrics
import custom_lgb


def cut_input(df, from_date, to_date):
    ret_df = df[(df['visit_date'] >= from_date) & (df['visit_date'] <= to_date)].reset_index(drop=True)
    return ret_df


# load data
train = pd.read_csv('../output/reg_train.csv')
predict = pd.read_csv('../output/reg_predict.csv')

# make input
train['visitors'] = np.log1p(train['visitors'])
# train_input = train[ (train['visit_date'] >= '2016-01-01') & (train['visit_date'] < '2016-11-28') ].reset_index(drop=True)
# test_input = train[ (train['visit_date'] >= '2017-01-16') & (train['visit_date'] < '2017-03-05') ].reset_index(drop=True)
# train_input = cut_input(train, "2016-01-16", "2016-11-28")
# test_input = cut_input(train, "2017-01-16", "2017-03-05")
train_input = cut_input(train, "2016-01-16", "2017-04-08")
test_input = cut_input(train, "2017-04-09", "2017-04-15")

col = ['air_store_num', 'visitors', 'air_genre_num', 'air_area_num', "prefecture_num", "city_num",
       'year', 'month', 'week',
       "moving_mean_0", "moving_median_0", "moving_max_0", "moving_min_0", "moving_std_0",
       "moving_mean_1", "moving_median_1", "moving_max_1", "moving_min_1", "moving_std_1",
       "moving_mean_3", "moving_median_3", "moving_max_3", "moving_min_3", "moving_std_3",
       "moving_mean_13", "moving_median_13", "moving_max_13", "moving_min_13", "moving_std_13",
       "dow_moving_mean_0", "dow_moving_median_0", "dow_moving_max_0", "dow_moving_min_0", "dow_moving_std_0",
       "dow_moving_mean_1", "dow_moving_median_1", "dow_moving_max_1", "dow_moving_min_1", "dow_moving_std_1",
       "dow_moving_mean_3", "dow_moving_median_3", "dow_moving_max_3", "dow_moving_min_3", "dow_moving_std_3",
       "dow_moving_mean_13", "dow_moving_median_13", "dow_moving_max_13", "dow_moving_min_13", "dow_moving_std_13",
       "change_mean_0_1", "change_mean_0_3", "change_mean_0_13", "change_mean_1_3", "change_mean_1_13", "change_mean_3_13",
       "dow_change_mean_0_1", "dow_change_mean_0_3", "dow_change_mean_0_13", "dow_change_mean_1_3", "dow_change_mean_1_13", "dow_change_mean_3_13",
       "change_median_0_1", "change_median_0_3", "change_median_0_13", "change_median_1_3", "change_median_1_13", "change_median_3_13",
       "dow_change_median_0_1", "dow_change_median_0_3", "dow_change_median_0_13", "dow_change_median_1_3", "dow_change_median_1_13", "dow_change_median_3_13",
       'dow', 'dowh', 'dows', 'holiday_flg', 'week_hols', 'next_week_hols', 'prev_week_hols',
       'quarter_regress', 'year_regress',
       "reserve_sum_air", "reserve_mean_air", "reserve_datediff_mean_air",
       "reserve_sum_hpg", "reserve_mean_hpg", "reserve_datediff_mean_hpg",
       # "total_reserve_sum", 'total_reserve_mean', 'total_reserve_dt_diff_mean',
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
y_pred = np.expm1(y_pred)
y_pred[y_pred < 1] = 1

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
fi = fi.sort_values(by="importance", ascending=False)
print(fi)

# submit
submission = pd.DataFrame({
        "id": predict['id'],
        "visitors": y_pred
    })
print(submission.describe())
submission.to_csv('../output/submission.csv',float_format='%.6f', index=False)

print("-"*30)
print("public_lb describe:")
submission["visit_date"] = submission['id'].map(lambda x: str(x).split('_')[2])
public_lb = submission[submission["visit_date"] <= "2017-04-28"]
print(public_lb.describe())
