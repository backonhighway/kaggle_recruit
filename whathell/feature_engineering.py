import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
from sklearn import linear_model
import time


def get_simple_grouped(df):
    grouped = df.groupby(["air_store_num", "year", "month"])\
        .agg({"visitors": ["min", "max", np.median, np.mean, np.std]}).reset_index()
    grouped.columns = ["air_store_num", "prev_month_y", "prev_month_m",
                       "min", "max", "median", "mean", "std"]

    return grouped


def get_dowh_grouped(df):
    grouped = df.groupby(["air_store_num", "year", "month", "dowh"])\
        .agg({"visitors": ["min", "max", np.median, np.mean, np.std]}).reset_index()
    grouped.columns = ["air_store_num", "prev_month_y", "prev_month_m", "dowh",
                       "min", "max", "median", "mean", "std"]

    return grouped


def get_stats(grouped):
    grouped["3month_min"] = grouped["min"].rolling(3, min_periods=1).min().reset_index(drop=True)
    grouped["3month_max"] = grouped["max"].rolling(3, min_periods=1).max().reset_index(drop=True)
    grouped["3month_median"] = grouped["median"].rolling(3, min_periods=1).median().reset_index(drop=True)
    grouped["3month_mean"] = grouped["mean"].rolling(3, min_periods=1).mean().reset_index(drop=True)
    grouped["3month_std"] = grouped["std"].rolling(3, min_periods=1).mean().reset_index(drop=True)
    grouped["6month_min"] = grouped["min"].rolling(6, min_periods=1).min().reset_index(drop=True)
    grouped["6month_max"] = grouped["max"].rolling(6, min_periods=1).max().reset_index(drop=True)
    grouped["6month_median"] = grouped["median"].rolling(6, min_periods=1).median().reset_index(drop=True)
    grouped["6month_mean"] = grouped["mean"].rolling(6, min_periods=1).mean().reset_index(drop=True)
    grouped["6month_std"] = grouped["std"].rolling(6, min_periods=1).mean().reset_index(drop=True)
    grouped["12month_min"] = grouped["min"].rolling(12, min_periods=1).min().reset_index(drop=True)
    grouped["12month_max"] = grouped["max"].rolling(12, min_periods=1).max().reset_index(drop=True)
    grouped["12month_median"] = grouped["median"].rolling(12, min_periods=1).median().reset_index(drop=True)
    grouped["12month_mean"] = grouped["mean"].rolling(12, min_periods=1).mean().reset_index(drop=True)
    grouped["12month_std"] = grouped["std"].rolling(12, min_periods=1).mean().reset_index(drop=True)

    return grouped


def engineer(df):
    df["visit_datetime"] = pd.to_datetime(df["visit_date"])

    df["dowh"] = np.where((df["holiday_flg"] == 1) & (df["dow"] < 5), 7, df["dow"])

    # df['visitors'] = np.log1p(df['visitors'])
    return df


# load data
train = pd.read_csv('../output/cleaned_res_train.csv')
predict = pd.read_csv('../output/cleaned_res_predict.csv')

print("loaded data.")

train = engineer(train)
predict = engineer(predict)


# set stats
print("setting stats...")
merge_col = ["air_store_num", "prev_month_y", "prev_month_m", "dowh"]
train_dowh_group = get_dowh_grouped(train)
train_dowh_group = get_stats(train_dowh_group)
train = pd.merge(train, train_dowh_group, how="left", on=merge_col)
predict_dowh_group = get_dowh_grouped(train)
predict_dowh_group = get_stats(predict_dowh_group)
predict = pd.merge(predict, predict_dowh_group, how="left", on=merge_col)

# print(train.head())
# print(predict.head())


print("output to csv...")
train.to_csv('../output/fed_train.csv',float_format='%.6f', index=False)
predict.to_csv('../output/fed_predict.csv',float_format='%.6f', index=False)

