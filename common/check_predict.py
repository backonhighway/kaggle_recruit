import pandas as pd
# load data
train = pd.read_csv('../output/w2_cwrrsr_train.csv')
predict = pd.read_csv('../output/w2_cwrrsr_predict.csv')

short_col = [
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
    #"air_r_sum0_shifted", "hpg_r_sum0_shifted", "air_dow_r_sum0_shifted", "hpg_dow_r_sum0_shifted",
    #"air_r_sum7", "hpg_r_sum7",
    #"air_r_date_diff_mean7", "hpg_r_date_diff_mean7",  # small
    #"air_r_date_diff_mean0_shifted", "hpg_r_date_diff_mean0_shifted",  # small
    #"air_r_sum7l", "hpg_r_sum7l", "total_r_sum7l",
]

predict = predict[(predict["visit_date"] >= "2017-04-29")& (predict["visit_date"] <= "2017-05-09")]
predict = predict[short_col]
#predict = predict[predict["air_store_num"] == 0]
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', None)
print(predict.groupby("visit_date").describe())