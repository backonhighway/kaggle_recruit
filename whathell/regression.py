from sklearn import linear_model
import pandas as pd
import numpy as np
import time
import pandas.tseries.offsets as offsets
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt


def take_df_by_period(df, timestamp_from, timestamp_to):
    time_from_str = timestamp_from.strftime('%Y-%m-%d')
    time_to_str = timestamp_to.strftime('%Y-%m-%d')
    return df[(df["visit_date"] >= time_from_str) & (df["visit_date"] <= time_to_str)]


def take_df_by_valid_period(df, timestamp_from, timestamp_to):
    if timestamp_to.year == 2017 and timestamp_to.month == 4:
        timestamp_to = timestamp_to.replace(day=22)
    return take_df_by_period(df, timestamp_from, timestamp_to)


def regress_by_store(df):
    ret_list = []
    month_ends = pd.date_range(start='01/01/2016', end='05/01/2017', freq='M')
    for month_end in month_ends:
        quarter_start = month_end - offsets.MonthBegin(3)
        quarter_df = take_df_by_valid_period(df, quarter_start, month_end)
        if quarter_df.empty:
            continue
        next_month_start = month_end + offsets.MonthBegin(1)
        next_month_end = month_end + offsets.MonthEnd(1)
        next_month_df = take_df_by_period(df, next_month_start, next_month_end)
        if next_month_df.empty:
            continue
        quarter_y_pred = do_regression(quarter_df, next_month_df)

        year_start = month_end - offsets.MonthBegin(12)
        year_df = take_df_by_valid_period(df, year_start, month_end)
        year_y_pred = do_regression(year_df, next_month_df)

        temp_df = pd.DataFrame(index=next_month_df.index)
        temp_df["quarter_regress"] = quarter_y_pred
        temp_df["year_regress"] = year_y_pred
        ret_list.append(temp_df)
    return ret_list


def do_regression(train_df, predict_df):
    x_train = pd.DataFrame(train_df["day_delta"])
    y_train = pd.DataFrame(train_df["visitors"])
    x_pred = pd.DataFrame(predict_df["day_delta"])
    reg = linear_model.Ridge(alpha=.5)
    y_pred = reg.fit(x_train, y_train).predict(x_pred)
    return y_pred


def regress(df):
    # col = ["air_store_num", "visitors", "visit_date", "dowh", "day_delta"]
    # df = org_df[col]
    start_time = time.time()
    result_df_list = []
    grouped = df.groupby(["air_store_num", "dows"])
    for name, group in grouped:
        if int(name[0]) % 10 == 0 and int(name[1]) == 0:
            print("air_store_num=", name[0])
            now_time = time.time()
            elapsed_time = now_time - start_time
            start_time = now_time
            print("elapsed_time=", elapsed_time)
        regressed = regress_by_store(group)
        if len(regressed) == 0:
            continue
        # merge_start = time.time()
        result_df_list.extend(regressed)
        # df = pd.merge(df, regressed, how="left")
        # merge_end = time.time()
        # print("merge_time is", merge_end - merge_start)
    conc = pd.concat(result_df_list)
    df = df.join(conc, how="left")
    return df


# load data
train = pd.read_csv('../output/cws_train.csv')
predict = pd.read_csv('../output/cws_predict.csv')

print("loaded data.")

# set regression
print("doing regression...")
joined = pd.concat([train, predict]).reset_index(drop=True)
# joined = joined[joined["air_store_num"] == 5]
joined = regress(joined)

# debug_df = joined[joined["visit_date"] >= "2017-04-01"]
# print(debug_df[["visit_date","quarter_regress"]].head(60))

print("done regression...")
train = joined[joined["visit_date"] < "2017-04-23"]
predict = joined[joined["visit_date"] >= "2017-04-23"]
# print(predict[["visit_date","quarter_regress"]].head(30))
# print(train.shape)
# print(train.isnull().sum())
# print(predict.shape)
# print(predict.isnull().sum())


print("output to csv...")
train.to_csv('../output/cwsr_train.csv',float_format='%.6f', index=False)
predict.to_csv('../output/cwsr_predict.csv',float_format='%.6f', index=False)
