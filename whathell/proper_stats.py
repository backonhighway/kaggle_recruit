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


def do_transforms(df, col_names):
    for col_name in col_names:
        df[col_name] = df.groupby("air_store_num")[col_name].transform(lambda x: x.shift(1))

    return df


def get_stats(df):
    col_names = ["moving_mean_1", "moving_median_1", "moving_max_1", "moving_min_1", "moving_std_1"]
    df = make_stats(df, 45, col_names)
    df = do_transforms(df, col_names)

    col_names = ["moving_mean_3", "moving_median_3", "moving_max_3", "moving_min_3", "moving_std_3"]
    df = make_stats(df, 90, col_names)
    df = do_transforms(df, col_names)

    col_names = ["moving_mean_13", "moving_median_13", "moving_max_13", "moving_min_13", "moving_std_13"]
    df = make_stats(df, 390, col_names)
    df = do_transforms(df, col_names)

    return df


def engineer(df):
    df["visit_datetime"] = pd.to_datetime(df["visit_date"])
    df["dowh"] = np.where((df["holiday_flg"] == 1) & (df["dow"] < 5), 7, df["dow"])
    df["dows"] = np.where(df["holiday_flg"] == 1, 6 - df["next_is_hol"], df["dow"])

    # df['visitors'] = np.log1p(df['visitors'])
    return df


# load data
train = pd.read_csv('../output/cleaned_res_train.csv')
predict = pd.read_csv('../output/cleaned_res_predict.csv')

print("loaded data.")
joined = pd.concat([train, predict]).reset_index(drop=True)
joined = engineer(joined)

# set stats
print("setting stats...")
joined = get_stats(joined)

train = joined[joined["visit_date"] < "2017-04-23"]
predict = joined[joined["visit_date"] >= "2017-04-23"]

print(train.head())
print(predict.head())


print("output to csv...")
train.to_csv('../output/proper_stats_train.csv',float_format='%.6f', index=False)
predict.to_csv('../output/proper_stats_predict.csv',float_format='%.6f', index=False)

