import numpy as np
import pandas as pd

import custom_metrics
import custom_lgb
import pocket_split_train
import pocket_full_of_validator
import pocket_ez_validator
import pocket_periods


# load data
train = pd.read_csv('../output/w2_cwrrsr_train.csv')
predict = pd.read_csv('../output/w2_cwrrsr_predict.csv')

# make input
#train = train[train["first_appear"] > "2016-03-05"]
#train = train[train["first_appear"] <= "2016-03-05"]
#train = train[train["visit_date"] >= "2016-06-01"]
train['visitors'] = np.log1p(train['visitors'])
# train_input = train[ (train['visit_date'] >= '2016-01-01') & (train['visit_date'] < '2016-11-28') ].reset_index(drop=True)
# test_input = train[ (train['visit_date'] >= '2017-01-16') & (train['visit_date'] < '2017-03-05') ].reset_index(drop=True)

col = [
    'air_store_num', 'visitors', 'air_genre_num', "air_area_num",
    'year', 'month', #"week",
    "moving_mean_0", "moving_median_0", "moving_max_0", "moving_std_0",
    #"moving_mean_05", "moving_median_05", "moving_max_05", "moving_std_05",
    "moving_mean_1", "moving_median_1", "moving_max_1", "moving_std_1",
    "moving_mean_3", "moving_median_3", "moving_max_3", "moving_std_3",
    "moving_mean_13", "moving_median_13", "moving_max_13", "moving_std_13",
    #"moving_min_0", "moving_min_1", "moving_min_3", "moving_min_13",  # small
    "dow_moving_mean_0", "dow_moving_median_0", "dow_moving_max_0", "dow_moving_std_0",
    #"dow_moving_mean_05", "dow_moving_median_05", "dow_moving_max_05", "dow_moving_std_05",
    "dow_moving_mean_1", "dow_moving_median_1", "dow_moving_max_1", "dow_moving_std_1",
    "dow_moving_mean_3", "dow_moving_median_3", "dow_moving_max_3", "dow_moving_std_3",
    "dow_moving_mean_13", "dow_moving_median_13", "dow_moving_max_13", "dow_moving_std_13",
    "change_mean_0_1", "change_mean_0_3", "change_mean_0_13",  # small
    "change_mean_1_3", "change_mean_1_13", "change_mean_3_13",  # small
    #"dow_change_mean_0_1", "dow_change_mean_0_3", "dow_change_mean_0_13",
    #"dow_change_mean_1_3", "dow_change_mean_1_13", "dow_change_mean_3_13",
    "moving_skew_0", "moving_skew_1", "moving_skew_3", "moving_skew_13",
    "dow_moving_skew_1", "dow_moving_skew_3", "dow_moving_skew_13",
    'dow', 'dowh', 'dows', 'holiday_flg', 'week_hols', 'next_week_hols', 'prev_week_hols', "next_is_hol",
    #'quarter_regress', 'year_regress', "ewm", "log_ewm", 'quarter_regress_no_dow', 'year_regress_no_dow',
    'quarter_regress', 'year_regress', "ewm", 'quarter_regress_no_dow', 'year_regress_no_dow',
    "precipitation", "avg_temperature",
    "air_r_sum0_shifted", "hpg_r_sum0_shifted", "air_dow_r_sum0_shifted", "hpg_dow_r_sum0_shifted",
    "air_r_sum7", "hpg_r_sum7",
    #"air_r_date_diff_mean7", "hpg_r_date_diff_mean7",  # small
    #"air_r_date_diff_mean0_shifted", "hpg_r_date_diff_mean0_shifted",  # small
    #"air_r_sum7l", "hpg_r_sum7l", "total_r_sum7l",
]


# fit and predict
# model = custom_lgb.fit(train_input, test_input)
period_list = [["2016-01-16", "2017-04-15", "2017-04-16", "2017-04-22"],
               #["2016-01-16", "2017-04-01", "2017-04-09", "2017-04-15"],
               #["2016-01-16", "2017-03-26", "2017-04-02", "2017-04-08"],
               #["2016-01-16", "2016-04-17", "2016-04-18", "2016-04-24"],
               #["2016-01-16", "2017-03-04", "2017-03-12", "2017-03-19"],
               ]
hmmm = train
hmmm = hmmm[(hmmm["visit_date"] <= "2016-05-02") | (hmmm["visit_date"] >= "2016-05-04")]
splits = pocket_split_train.split_set(hmmm, period_list, col)
#splits = pocket_split_train.split_set(train, period_list, col)
models = pocket_split_train.split_train(splits)
model = models[0]
print("-" * 40)
pocket_full_of_validator.validate(train, model, col, "2016-05-03", "2016-05-03")
pocket_full_of_validator.validate(train, model, col, "2016-04-28", "2016-04-28")
pocket_full_of_validator.validate(train, model, col, "2016-04-29", "2016-04-29")
pocket_full_of_validator.validate(train, model, col, "2016-04-30", "2016-04-30")
print("-" * 40)
pocket_full_of_validator.validate(train, model, col, "2016-05-01", "2016-05-01")
pocket_full_of_validator.validate(train, model, col, "2016-05-02", "2016-05-02")
pocket_full_of_validator.validate(train, model, col, "2016-05-03", "2016-05-03")
pocket_full_of_validator.validate(train, model, col, "2016-05-04", "2016-05-04")
pocket_full_of_validator.validate(train, model, col, "2016-05-05", "2016-05-05")
pocket_full_of_validator.validate(train, model, col, "2016-05-05", "2016-05-06")
pocket_full_of_validator.validate(train, model, col, "2016-05-06", "2016-05-07")
pocket_full_of_validator.validate(train, model, col, "2016-05-07", "2016-05-08")
pocket_full_of_validator.validate(train, model, col, "2016-05-08", "2016-05-09")

exit(0)
