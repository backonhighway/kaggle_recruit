import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
import pandas.tseries.offsets as offsets
from sklearn import linear_model
import time


def make_stats(df, window_days, col_names):
    rolling = df.groupby("air_store_num")["visitors"].rolling(window_days, min_periods=0)
    df[col_names[0]] = rolling.mean().reset_index(0, drop=True)
    df[col_names[1]] = rolling.median().reset_index(0, drop=True)
    df[col_names[2]] = rolling.max().reset_index(0, drop=True)
    df[col_names[3]] = rolling.min().reset_index(0, drop=True)
    df[col_names[4]] = rolling.std().reset_index(0, drop=True)
    return df


def make_dow_stats(df, window_days, col_names):
    grouped = df.groupby(["air_store_num", "dows"])["visitors"]
    df[col_names[0]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).mean().shift(1))
    df[col_names[1]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).median().shift(1))
    df[col_names[2]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).max().shift(1))
    df[col_names[3]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).min().shift(1))
    df[col_names[4]] = grouped.transform(lambda x: x.rolling(window_days, min_periods=0).std().shift(1))
    return df


def do_transforms(df, col_names):
    for col_name in col_names:
        df[col_name] = df.groupby("air_store_num")[col_name].transform(lambda x: x.shift(1))

    return df


def get_stats(df):
    # Should not use 0 after second week
    col_names = ["moving_mean_0", "moving_median_0", "moving_max_0", "moving_min_0", "moving_std_0"]
    df = make_stats(df, 7, col_names)
    df = do_transforms(df, col_names)
    col_names = ["moving_mean_1", "moving_median_1", "moving_max_1", "moving_min_1", "moving_std_1"]
    df = make_stats(df, 45, col_names)
    df = do_transforms(df, col_names)
    col_names = ["moving_mean_3", "moving_median_3", "moving_max_3", "moving_min_3", "moving_std_3"]
    df = make_stats(df, 90, col_names)
    df = do_transforms(df, col_names)
    col_names = ["moving_mean_13", "moving_median_13", "moving_max_13", "moving_min_13", "moving_std_13"]
    df = make_stats(df, 390, col_names)
    df = do_transforms(df, col_names)

    col_names = ["dow_moving_mean_0", "dow_moving_median_0", "dow_moving_max_0", "dow_moving_min_0", "dow_moving_std_0"]
    df = make_dow_stats(df, 1, col_names)
    col_names = ["dow_moving_mean_1", "dow_moving_median_1", "dow_moving_max_1", "dow_moving_min_1", "dow_moving_std_1"]
    df = make_dow_stats(df, 5, col_names)
    col_names = ["dow_moving_mean_3", "dow_moving_median_3", "dow_moving_max_3", "dow_moving_min_3", "dow_moving_std_3"]
    df = make_dow_stats(df, 15, col_names)
    col_names = ["dow_moving_mean_13", "dow_moving_median_13", "dow_moving_max_13", "dow_moving_min_13", "dow_moving_std_13"]
    df = make_dow_stats(df, 55, col_names)
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
    grouped = df.groupby(["air_store_num", "dows"])["visitors"]
    df["ewm"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(1))
    return df


# load data
train = pd.read_csv('../output/cw_train.csv')
predict = pd.read_csv('../output/cw_predict.csv')
predict["visitors"] = np.NaN

print("loaded data.")
joined = pd.concat([train, predict]).reset_index(drop=True)

# joined = joined[joined["air_store_num"] == 4]
# df = df[["air_store_num", "visitors", "visit_date", "ewm"]]
# print(df.tail(90))

# set stats
print("setting stats...")
joined = get_ewm(joined)
joined = get_stats(joined)
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
train.to_csv('../output/cws_train.csv',float_format='%.6f', index=False)
predict.to_csv('../output/cws_predict.csv',float_format='%.6f', index=False)

