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


def take_df_by_period(df, timestamp_from, timestamp_to):
    time_from_str = timestamp_from.strftime('%Y-%m-%d')
    time_to_str = timestamp_to.strftime('%Y-%m-%d')
    return df[(df["visit_date"] >= time_from_str) & (df["visit_date"] <= time_to_str)]


def regress_by_store(df):
    ret_df = pd.DataFrame({})
    month_ends = pd.date_range(start='01/01/2016', end='04/01/2017', freq='M')
    for month_end in month_ends:
        quarter_start = month_end - offsets.MonthBegin(3)
        quarter_df = take_df_by_period(df, quarter_start, month_end)
        if quarter_df.empty:
            continue
        next_month_start = month_end + offsets.MonthBegin(1)
        next_month_end = month_end + offsets.MonthEnd(1)
        next_month_df = take_df_by_period(df, next_month_start, next_month_end)
        x_train = pd.DataFrame(quarter_df["day_delta"])
        y_train = pd.DataFrame(quarter_df["visitors"])
        x_pred = pd.DataFrame(next_month_df["day_delta"])
        if x_pred.empty:
            continue
        reg = linear_model.Ridge(alpha=.5)
        y_pred = reg.fit(x_train, y_train).predict(x_pred)
        x_pred["q_pred"] = y_pred

        #year_start = month_end - offsets.MonthBegin(12)
        #year_df = take_df_by_period(df, year_start, month_end)
        #x_train = pd.DataFrame(year_df["day_delta"])
        #y_train = pd.DataFrame(year_df["visitors"])
        #print(x_train.shape)
        #print(y_train.shape)
        #reg = linear_model.Ridge(alpha=.5)
        #y_pred = reg.fit(x_train, y_train).predict(x_pred)
        #x_pred["y_pred"] = y_pred
        ret_df = ret_df.append(x_pred)
    return ret_df


def do_regression(train_df, predict_df):
    x_train = pd.DataFrame(train_df["day_delta"])
    y_train = pd.DataFrame(train_df["visitors"])
    x_pred = pd.DataFrame(predict_df["day_delta"])
    reg = linear_model.Ridge(alpha=.5)
    y_pred = reg.fit(x_train, y_train).predict(x_pred)
    return y_pred


def regress(df):
    start_time = time.time()
    grouped = df.groupby(["air_store_num"])
    for name, group in grouped:
        #if int(name[0]) % 10 == 0 and int(name[1]) == 0:
        if int(name) % 10 == 0:
            print("air_store_num=", name)
            now_time = time.time()
            elapsed_time = now_time - start_time
            start_time = now_time
            print("elapsed_time=", elapsed_time)
        regressed = regress_by_store(group)
        if regressed.empty:
            continue
        merge_start_time = time.time()
        #df = pd.merge(df, regressed, how="left", on="day_delta")
        print(df.index)
        print(regressed.index)
        df.join(regressed, how="outer")
        merge_end_time = time.time()
        print("merge_time is", merge_end_time - merge_start_time)
    return df


def engineer(df):
    df["visit_datetime"] = pd.to_datetime(df["visit_date"])

    df["dowh"] = np.where((df["holiday_flg"] == 1) & (df["dow"] < 5), 7, df["dow"])

    # df['visitors'] = np.log1p(df['visitors'])
    return df


# load data
train = pd.read_csv('../output/cleaned_train.csv')
predict = pd.read_csv('../output/cleaned_predict.csv')

print("loaded data.")

train = engineer(train)
predict = engineer(predict)

# set regression
print("doing regression...")
train = regress(train)
predict = regress(predict)


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

