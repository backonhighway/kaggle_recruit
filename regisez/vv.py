import pandas as pd
import numpy as np
from sklearn import linear_model
import time
import pandas.tseries.offsets as offsets
import pocket_ez_validator


train = pd.read_csv('../output/cwsvr_train.csv')
print("Loaded data.")

use_col = ["air_store_num", "dow", "visitors", "visit_date", "dows", "day_delta"]
train = train[use_col]
train["log_visitors"] = np.log1p(train["visitors"])
last_train_date = "2017-03-18"
train["log_visitors_nan"] = np.where(train["visit_date"] <= last_train_date, train["log_visitors"], np.NaN)
train["visitors_nan"] = np.where(train["visit_date"] <= last_train_date, train["visitors"], np.NaN)
# train = train[train["air_store_num"] <= 1]


def take_df_by_period(df: pd.DataFrame, timestamp_from, timestamp_to):
    time_from_str = timestamp_from.strftime('%Y-%m-%d')
    time_to_str = timestamp_to.strftime('%Y-%m-%d')
    return df[(df["visit_date"] >= time_from_str) & (df["visit_date"] <= time_to_str)]


def do_regression(train_df, predict_df, models):
    pred_list = []
    x_train = pd.DataFrame(train_df["day_delta"])
    y_train = pd.DataFrame(train_df["log_visitors"])
    x_pred = pd.DataFrame(predict_df["day_delta"])
    for model in models:
        y_pred = model.fit(x_train, y_train).predict(x_pred)
        pred_list.append(y_pred)
    return pred_list


def regress_em(df):
    ret_list = []
    predict_start = pd.to_datetime("2017-03-18", infer_datetime_format='%Y-%m-%d')
    predict_end = pd.to_datetime("2017-04-22", infer_datetime_format='%Y-%m-%d')
    train_end = predict_start - offsets.Day(1)
    quarter_start = train_end - offsets.Week(13)
    year_start = train_end - offsets.Week(52)

    predict_df = take_df_by_period(df, predict_start, predict_end)
    quarter_df = take_df_by_period(df, quarter_start, train_end).dropna(axis=0, subset=["visitors_nan"])
    year_df = take_df_by_period(df, year_start, train_end).dropna(axis=0, subset=["visitors_nan"])
    if predict_df.empty or quarter_df.empty:
        return ret_list

    li1 = linear_model.LinearRegression()
    r1 = linear_model.Ridge(alpha=0.1)
    r2 = linear_model.Ridge(alpha=0.5)
    r3 = linear_model.Ridge(alpha=1.0)
    l1 = linear_model.Lasso(alpha=0.1)
    l2 = linear_model.Lasso(alpha=0.5)
    l3 = linear_model.Lasso(alpha=1.0)
    # h1 = linear_model.HuberRegressor()
    models = [li1, r1, r2, r3, l1, l2, l3]
    quarter_y_pred_list = do_regression(quarter_df, predict_df, models)
    year_y_pred_list = do_regression(year_df, predict_df, models)
    name_list = ["li1", "r1", "r2", "r3", "l1", "l2", "l3"]
    temp_df = pd.DataFrame(index=predict_df.index)
    for i in range(7):
        col_name = "q_regress_" + name_list[i]
        temp_df[col_name] = quarter_y_pred_list[i]
        col_name = "y_regress_" + name_list[i]
        temp_df[col_name] = year_y_pred_list[i]
    ret_list.append(temp_df)
    return ret_list


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
        regressed = regress_em(group)
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


train = regress(train)
# train["year_regress"] = np.log1p(train["year_regress"])
# train["quarter_regress"] = np.log1p(train["quarter_regress"])
train = train[train["visit_date"] >= "2017-03-19"].dropna(axis=0, subset=["q_regress_li1", "y_regress_li1"])

name_list = ["li1", "r1", "r2", "r3", "l1", "l2", "l3"]
for i in range(7):
    col_name = "q_regress_" + name_list[i]
    #train[col_name] = np.log1p(train[col_name]).dropna(axis=0, subset=[col_name])
    pocket_ez_validator.validate(train, "log_visitors", col_name)

for i in range(7):
    col_name = "y_regress_" + name_list[i]
    #train[col_name] = np.log1p(train[col_name]).dropna(axis=0, subset=[col_name])
    pocket_ez_validator.validate(train, "log_visitors", col_name)