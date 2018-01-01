import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets


def get_stats_group(df):
    grouped = df.groupby(["air_store_num", "year", "month"])\
        .agg({"visitors": ["min", "max", np.median, np.mean, np.std]}).reset_index()
    grouped.columns = ["air_store_num", "prev_month_y", "prev_month_m", "min", "max", "median", "mean", "std"]

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
    df["year"] = df["visit_datetime"].dt.year
    df["month"] = df["visit_datetime"].dt.month
    df["prev_month"] = df["visit_datetime"] - offsets.MonthBegin(2)
    df["prev_month_y"] = df["prev_month"].dt.year
    df["prev_month_m"] = df["prev_month"].dt.month

    # df["monthly_mean"] = df.groupby(["air_store_num", "month"])["visitors"].transform(np.mean)
    # df["monthly_median"] = df.groupby(["air_store_num", "month"])["visitors"].transform(np.median)

    # df['visitors'] = np.log1p(df['visitors'])

    return df


# load data
train = pd.read_csv('../output/cleaned_train.csv')
predict = pd.read_csv('../output/cleaned_predict.csv')

# print(train.head())
# print(predict.head())
print('-' * 50)

train = engineer(train)
predict = engineer(predict)

stats_group = get_stats_group(train)
train = pd.merge(train, stats_group, how="left", on=["air_store_num", "prev_month_y", "prev_month_m"])
predict = pd.merge(predict, stats_group, how="left", on=["air_store_num", "prev_month_y", "prev_month_m"])

# print(train.head())
# print(predict.head())
# exit(0)

train.to_csv('../output/fed_train.csv',float_format='%.6f', index=False)
predict.to_csv('../output/fed_predict.csv',float_format='%.6f', index=False)

