import pandas as pd
import numpy as np


train = pd.read_csv('../output/w2_cwrrsr_train.csv')
predict = pd.read_csv('../output/w2_cwrrsr_predict.csv')

#train['visitors'] = np.log1p(train['visitors'])
#train = train[train["air_store_num"] < 2]
use_col = [
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
    "change_mean_0_1", "change_mean_0_3", "change_mean_0_13",  # small
    "change_mean_1_3", "change_mean_1_13", "change_mean_3_13",  # small
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

train = train[pd.notnull(train["moving_mean_0"])]
train = train[pd.notnull(train["dow_moving_skew_1"])]

reserve_col = [
    "air_r_sum0_shifted", "hpg_r_sum0_shifted", "air_dow_r_sum0_shifted", "hpg_dow_r_sum0_shifted",
    "air_r_sum7", "hpg_r_sum7",
]
train[reserve_col] = train[reserve_col].fillna(0)

#cols = list(train.columns)
for col in use_col:
    #col_zscore = col + '_zscore'
    train[col] = (train[col] - train[col].mean())/train[col].std(ddof=0)
use_col.extend(["visit_date", "visitors"])
train = train[use_col]

pd.set_option('display.max_columns', None)
print(train.head())

train = train.fillna(0)
predict = predict.fillna(0)

train.to_csv('../output/z_train_w2.csv',float_format='%.6f', index=False)
#predict.to_csv('../output/z_predict_w2.csv',float_format='%.6f', index=False)
