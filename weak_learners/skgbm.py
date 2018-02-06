import numpy as np
import pandas as pd

import custom_metrics
import custom_lgb
import pocket_split_train
import pocket_full_of_validator
import pocket_ez_validator
import pocket_periods
from sklearn import neighbors, ensemble, linear_model
import time


# load data
train = pd.read_csv('../output/z_train_w2.csv')
predict = pd.read_csv('../output/w2_cwrrsr_predict.csv')

train['visitors'] = np.log1p(train['visitors'])

# make input
col = [
    #'air_store_num', 'visitors', 'air_genre_num', "air_area_num",
    #'year', 'month', #"week",
    "moving_mean_0", "moving_median_0", "moving_max_0", "moving_std_0",
    "moving_mean_1", "moving_median_1", "moving_max_1", "moving_std_1",
    "moving_mean_3", "moving_median_3", "moving_max_3", "moving_std_3",
    "moving_mean_13", "moving_median_13", "moving_max_13", "moving_std_13",
    #"moving_min_0", "moving_min_1", "moving_min_3", "moving_min_13",  # small
    #"dow_moving_mean_0", "dow_moving_median_0", "dow_moving_max_0",
    "dow_moving_mean_1", "dow_moving_median_1", "dow_moving_max_1", "dow_moving_std_1",
    "dow_moving_mean_3", "dow_moving_median_3", "dow_moving_max_3", "dow_moving_std_3",
    "dow_moving_mean_13", "dow_moving_median_13", "dow_moving_max_13", "dow_moving_std_13",
    #"change_mean_0_1", "change_mean_0_3", "change_mean_0_13",  # small
    #"change_mean_1_3", "change_mean_1_13", "change_mean_3_13",  # small
    "moving_skew_0", "moving_skew_1", "moving_skew_3", "moving_skew_13",
    "dow_moving_skew_1", "dow_moving_skew_3", "dow_moving_skew_13",
    'holiday_flg', 'week_hols', 'next_week_hols', 'prev_week_hols', "next_is_hol",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    #'dow', 'dowh', 'dows', 'holiday_flg', 'week_hols', 'next_week_hols', 'prev_week_hols', "next_is_hol",
    'quarter_regress', 'year_regress', "ewm", 'quarter_regress_no_dow', 'year_regress_no_dow',
    #"precipitation", "avg_temperature",
    "air_r_sum0_shifted", "hpg_r_sum0_shifted", "air_dow_r_sum0_shifted", "hpg_dow_r_sum0_shifted",
    "air_r_sum7", "hpg_r_sum7",
]
#print(train["air_area_num"].value_counts())
#pd.set_option('display.max_columns', None)
#print(train.head())
#exit(0)


# fit and predict
# model = custom_lgb.fit(train_input, test_input)
period_list = [["2016-01-16", "2017-04-08", "2017-04-16", "2017-04-22"],
               ["2016-01-16", "2017-04-01", "2017-04-09", "2017-04-15"],
               ["2016-01-16", "2017-03-26", "2017-04-02", "2017-04-08"],
               #["2016-01-16", "2016-04-17", "2016-04-18", "2016-04-24"],
               ["2016-01-16", "2017-03-04", "2017-03-12", "2017-03-19"],
               ]
start_time = time.time()
model1 = ensemble.GradientBoostingRegressor(learning_rate=0.5, random_state=3)
model2 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
model3 = linear_model.Ridge(alpha=1.0)
model4 = linear_model.LinearRegression()
model5 = linear_model.Lasso(alpha=0.5)
#pocket_split_train.split_train_skt(train, col, period_list, model1)
#pocket_split_train.split_train_skt(train, col, period_list, model2)
pocket_split_train.split_train_skt(train, col, period_list, model3)
print("-"*40)
pocket_split_train.split_train_skt(train, col, period_list, model4)
print("-"*40)
pocket_split_train.split_train_skt(train, col, period_list, model5)
print("-"*40)
now_time = time.time()
elapsed_time = now_time - start_time
print("elapsed_time=", elapsed_time)
print("-" * 40)
exit(0)
fi = model1.feature_importances_
print(fi)

'''
# train.drop("visit_date", axis=1, inplace=True)
predict.drop("visit_date", axis=1, inplace=True)

x_pred = predict[col].drop('visitors', axis=1)
y_pred = model.predict(x_pred, num_iteration=model.best_iteration)
y_pred = np.expm1(y_pred)
y_pred[y_pred < 1] = 1


def save_models():
    suffix="_no_res.csv"
    pocket_full_of_validator.validate(train, models[0], col, "2017-04-16", "2017-04-22",
                                      save_model=True, save_name="../output/p1_w2_0416_0422"+suffix)
    pocket_full_of_validator.validate(train, models[1], col, "2017-04-09", "2017-04-15",
                                      save_model=True, save_name="../output/p1_w2_0409_0415"+suffix)
    pocket_full_of_validator.validate(train, models[2], col, "2017-04-02", "2017-04-08",
                                      save_model=True, save_name="../output/p1_w2_0402_0408"+suffix)
    pocket_full_of_validator.validate(train, models[3], col, "2017-03-12", "2017-03-19",
                                      save_model=True, save_name="../output/p1_w2_0312_0319"+suffix)



# save_models()
print("Validation score by six week:")
pocket_full_of_validator.validate(train, model, col, "2017-03-05", "2017-04-22")
print("Validation score by last week:")
pocket_full_of_validator.validate(train, model, col, "2017-04-16", "2017-04-22")
print("week by week")
for period in pocket_periods.get_six_week_period_list():
    pocket_full_of_validator.validate(train, model, col, period[0], period[1], True)




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
print("golden week describe:")
golden_week = submission[(submission["visit_date"] >= "2017-04-29") & (submission["visit_date"] <= "2017-05-07")]
print(golden_week.describe())
print(golden_week.groupby("visit_date").describe())

'''


