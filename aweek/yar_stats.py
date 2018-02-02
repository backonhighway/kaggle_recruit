import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
import pandas.tseries.offsets as offsets
from sklearn import linear_model
import time
import lolpy


SHIFT_WEEKS = 5


def make_ez_stats(df, window_days, dow_window_days, suffix):
    col_names = ["moving_mean", "moving_median", "moving_max", "moving_min",
                 "moving_std",
                 "moving_skew", "moving_kurt"]
    col_names = [s + suffix for s in col_names]
    grouped = df.groupby("air_store_num")["visitors_nan"]
    df = get_ez_stats(df, grouped, window_days, col_names, SHIFT_WEEKS * 7)

    col_names = ["dow_" + s for s in col_names]
    grouped = df.groupby(["air_store_num", "dow"])["visitors_nan"]
    df = get_ez_stats(df, grouped, dow_window_days, col_names, SHIFT_WEEKS)
    return df


def get_ez_stats(df, grouped, window_days, col_names, shift_days):
    df[col_names[0]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).mean().shift(shift_days))
    df[col_names[1]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).median().shift(shift_days))
    df[col_names[2]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).max().shift(shift_days))
    df[col_names[3]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).min().shift(shift_days))
    df[col_names[4]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).std().shift(shift_days))
    if window_days > 3:
        df[col_names[5]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).skew().shift(shift_days))
        df[col_names[6]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).kurt().shift(shift_days))
    return df


def get_stats(df):
    df = make_ez_stats(df, 7, 1, "_0")
    df = make_ez_stats(df, 45, 5, "_1")
    df = make_ez_stats(df, 90, 15, "_3")
    df = make_ez_stats(df, 390, 55, "_13")
    return df


def do_ez_filling(df):
    col_names = ["moving_mean", "moving_median", "moving_max", "moving_min",
                 "moving_std",
                 "moving_skew", "moving_kurt"]
    suffixes = ["_0", "_1", "_3", "_13"]
    for suffix in suffixes:
        col_name_list = [s + suffix for s in col_names]
        df = get_filled_rolling(df, col_name_list)
        col_name_list = ["dow_" + s + suffix for s in col_names]
        lolpy.remove_if_exists(col_name_list, ["dow_moving_skew_0", "dow_moving_kurt_0"])
        df = get_filled_rolling(df, col_name_list)
    return df


def get_filled_rolling(df: pd.DataFrame, col_name_list):
    for col_name in col_name_list:
        df[col_name] = np.where(df["visit_date"] <= "2017-03-04", df[col_name], np.NaN)
        grouped = df.groupby(["air_store_num", "dow"])[col_name]
        df[col_name] = grouped.transform(lambda x: x.fillna(method="ffill"))
    return df


def get_int_percentage(colA, colB):
    return (colA / colB).multiply(100).round()


def get_change_rate(df):
    df["change_mean_0_1"] = get_int_percentage(df["moving_mean_0"], df["moving_mean_1"])
    df["change_mean_0_3"] = get_int_percentage(df["moving_mean_0"], df["moving_mean_3"])
    df["change_mean_0_13"] = get_int_percentage(df["moving_mean_0"], df["moving_mean_13"])
    df["change_mean_1_3"] = get_int_percentage(df["moving_mean_1"], df["moving_mean_3"])
    df["change_mean_1_13"] = get_int_percentage(df["moving_mean_1"], df["moving_mean_13"])
    df["change_mean_3_13"] = get_int_percentage(df["moving_mean_3"], df["moving_mean_13"])
    df["dow_change_mean_0_1"] = get_int_percentage(df["dow_moving_mean_0"], df["dow_moving_mean_1"])
    df["dow_change_mean_0_3"] = get_int_percentage(df["dow_moving_mean_0"], df["dow_moving_mean_3"])
    df["dow_change_mean_0_13"] = get_int_percentage(df["dow_moving_mean_0"], df["dow_moving_mean_13"])
    df["dow_change_mean_1_3"] = get_int_percentage(df["dow_moving_mean_1"], df["dow_moving_mean_3"])
    df["dow_change_mean_1_13"] = get_int_percentage(df["dow_moving_mean_1"], df["dow_moving_mean_13"])
    df["dow_change_mean_3_13"] = get_int_percentage(df["dow_moving_mean_3"], df["dow_moving_mean_13"])
    df["change_median_0_1"] = get_int_percentage(df["moving_median_0"], df["moving_median_1"])
    df["change_median_0_3"] = get_int_percentage(df["moving_median_0"], df["moving_median_3"])
    df["change_median_0_13"] = get_int_percentage(df["moving_median_0"], df["moving_median_13"])
    df["change_median_1_3"] = get_int_percentage(df["moving_median_1"], df["moving_median_3"])
    df["change_median_1_13"] = get_int_percentage(df["moving_median_1"], df["moving_median_13"])
    df["change_median_3_13"] = get_int_percentage(df["moving_median_3"], df["moving_median_13"])
    df["dow_change_median_0_1"] = get_int_percentage(df["dow_moving_median_0"], df["dow_moving_median_1"])
    df["dow_change_median_0_3"] = get_int_percentage(df["dow_moving_median_0"], df["dow_moving_median_3"])
    df["dow_change_median_0_13"] = get_int_percentage(df["dow_moving_median_0"], df["dow_moving_median_13"])
    df["dow_change_median_1_3"] = get_int_percentage(df["dow_moving_median_1"], df["dow_moving_median_3"])
    df["dow_change_median_1_13"] = get_int_percentage(df["dow_moving_median_1"], df["dow_moving_median_13"])
    df["dow_change_median_3_13"] = get_int_percentage(df["dow_moving_median_3"], df["dow_moving_median_13"])


def calc_ewm(series, alpha, adjust=True):
    return series.ewm(alpha=alpha, adjust=adjust).mean()


def get_ewm(df):
    grouped = df.groupby(["air_store_num", "dow"])["visitors"]
    df["ewm"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(SHIFT_WEEKS))
    grouped = df.groupby(["air_store_num", "dow"])["log_visitors"]
    df["log_ewm"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(SHIFT_WEEKS))
    return df


# load data
train = pd.read_csv('../output/all_cwrr_train.csv')
predict = pd.read_csv('../output/all_cwrr_predict.csv')
predict["visitors"] = np.NaN

print("loaded data.")
joined = pd.concat([train, predict]).reset_index(drop=True)
joined["visitors_nan"] = np.where(joined["visit_date"] <= "2017-04-08", joined["visitors"], np.NaN)
joined["log_visitors_nan"] = np.where(joined["visit_date"] <= "2017-04-08", joined["log_visitors"], np.NaN)


# joined = joined[joined["air_store_num"] <= 0]
# df = df[["air_store_num", "visitors", "visit_date", "ewm"]]
# print(df.tail(90))

# set stats
print("setting stats...")
joined = get_ewm(joined)
joined = get_stats(joined)
#TODO the shift jumps over closed days
#pd.set_option("display.max_columns", 101)
#joined = joined[joined["visit_date"] >= "2017-04-01"]
#joined = joined[joined["visit_date"] <= "2017-05-05"]
#print(joined)
#exit(0)

print("doing filling and change rate")
#joined = do_ez_filling(joined)
get_change_rate(joined)

# pd.set_option('display.width', 240)
# pd.set_option('display.max_columns', None)
# print(joined.tail(30))
# exit(0)

train = joined[joined["visit_date"] < "2017-04-23"]
predict = joined[joined["visit_date"] >= "2017-04-23"]
# predict["visitors"] = 0

# print(train.head(10))
# print(predict.head())

print("output to csv...")
train.to_csv('../output/w5_cwrrs_train.csv',float_format='%.6f', index=False)
predict.to_csv('../output/w5_cwrrs_predict.csv',float_format='%.6f', index=False)

