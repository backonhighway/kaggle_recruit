import pandas as pd
import numpy as np
import pocket_ez_validator
import pocket_periods


train = pd.read_csv('../output/cwsvr_train.csv')
print("Loaded data.")
use_col = ["air_store_num", "dow", "visitors", "visit_date", "dows"]
train = train[use_col]
train["log_visitors"] = np.log1p(train["visitors"])
train["log_visitors_nan"] = np.where(train["visit_date"] <= "2017-03-04", train["log_visitors"], np.NaN)
train["visitors_nan"] = np.where(train["visit_date"] <= "2017-03-04", train["visitors"], np.NaN)
# train = train[train["air_store_num"] <= 3]

for weekly_df in pocket_periods.get_six_week_df_list(train):
    print(weekly_df["log_visitors"].describe())

# 71, 453, 527 are from 03/02, 03/02, 03/07. 677 has only 02/15 then 03/02
# g = train.groupby("air_store_num")["visit_date"].apply(np.min).reset_index(name="minn")
# g = g.sort_values(by="minn")
# print(g)
# exit (0)

# train = train[train["air_store_num"].isin([71, 453, 527, 677]) == False]
# train = train[train["air_store_num"].isin([826, 265, 335, 701, 202, 379]) == False]


#  mean and median
train["mean_visitors"] = train.groupby(["air_store_num", "dow"])["visitors_nan"].transform(np.mean)
train["mean_visitors"] = np.log1p(train["mean_visitors"])
train["log_mean_visitors"] = train.groupby(["air_store_num", "dow"])["log_visitors_nan"].transform(np.mean)
train["log_median_visitors"] = train.groupby(["air_store_num", "dow"])["log_visitors_nan"].transform(np.median)
# oh_wow = train[train["mean_visitors"].isnull()]
# print(oh_wow)
# exit(0)

train.dropna(axis=0, subset=["mean_visitors"], inplace=True)
# print(train.head(30))
print("Validate mean...")
pocket_ez_validator.validate(train, "log_visitors", "mean_visitors")
print("Validate log_mean...")
pocket_ez_validator.validate(train, "log_visitors", "log_mean_visitors")
print("Validate log_median...")
pocket_ez_validator.validate(train, "log_visitors", "log_median_visitors")


def calc_ewm(series, alpha, adjust=True):
    return series.ewm(alpha=alpha, adjust=adjust).mean()


def get_ewm(df):
    # delete 417, 736, 447 for dows
    # grouped = df.groupby(["air_store_num", "dows"])["log_visitors_nan"]
    grouped = df.groupby(["air_store_num", "dow"])["log_visitors_nan"]
    df["log_ewm"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(1))
    grouped = df.groupby(["air_store_num", "dow"])["visitors_nan"]
    df["ewm"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(1))
    return df


# ewm
print("Validate ewm...")
train = get_ewm(train)
# train = train[train["air_store_num"].isin([417, 736, 447]) == False]
pocket_ez_validator.validate(train, "log_visitors", "log_ewm", verbose=True)
train["ewm"] = np.log1p(train["ewm"])
pocket_ez_validator.validate(train, "log_visitors", "ewm")


# rolling average
def get_rolling_means(df):
    grouped = df.groupby(["air_store_num", "dow"])["log_visitors_nan"]
    df["dow_roll_mean1"] = grouped.transform(lambda x: x.rolling(1, min_periods=0).mean().shift(1).fillna(method="ffill"))
    df["dow_roll_mean5"] = grouped.transform(lambda x: x.rolling(5, min_periods=0).mean().shift(1).fillna(method="ffill"))
    df["dow_roll_mean15"] = grouped.transform(lambda x: x.rolling(15, min_periods=0).mean().shift(1).fillna(method="ffill"))
    df["dow_roll_mean55"] = grouped.transform(lambda x: x.rolling(55, min_periods=0).mean().shift(1).fillna(method="ffill"))
    return df


def get_filled_rolling(df: pd.DataFrame, col_name_list):
    for col_name in col_name_list:
        next_col_name = col_name + "_nan"
        df[next_col_name] = np.where(train["visit_date"] <= "2017-03-04", train[col_name], np.NaN)
        grouped = df.groupby(["air_store_num", "dow"])[next_col_name]
        df[next_col_name] = grouped.transform(lambda x: x.fillna(method="ffill"))
    return df


print("Validate rolling...")
col_list = ["dow_roll_mean1", "dow_roll_mean5", "dow_roll_mean15", "dow_roll_mean55"]
train = get_rolling_means(train)
train = get_filled_rolling(train, col_list)

train.dropna(axis=0, subset=["dow_roll_mean1_nan"], inplace=True)
print(train["air_store_num"].nunique())

# print(train.tail(20))
for col in col_list:
    pocket_ez_validator.validate(train, "log_visitors", col)
    next_col = col + "_nan"
    pocket_ez_validator.validate(train, "log_visitors", next_col)









