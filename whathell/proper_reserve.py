import pandas as pd
import numpy as np


def prepare(df):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
    df['visit_date'] = df['visit_datetime'].dt.date
    df["dow"] = df["visit_datetime"].dt.dayofweek
    df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
    df['reserve_date'] = df['reserve_datetime'].dt.date
    df['reserve_datetime_diff'] = df.apply(lambda r: (r['visit_date'] - r['reserve_date']).days, axis=1)
    return df


def get_dow_reserve_sum(df):
    if df["dow"] == 0:
        return df["sum2"]
    elif df["dow"] == 1:
        return df["sum3"]
    elif df["dow"] == 2:
        return df["sum4"]
    elif df["dow"] == 3:
        return df["sum5"]
    elif df["dow"] == 4:
        return df["sum6"]
    else:
        return df["sum1"]
    
    
def get_dow_reserve_mean(df):
    if df["dow"] == 0:
        return df["mean2"]
    elif df["dow"] == 1:
        return df["mean3"]
    elif df["dow"] == 2:
        return df["mean4"]
    elif df["dow"] == 3:
        return df["mean5"]
    elif df["dow"] == 4:
        return df["mean6"]
    else:
        return df["mean1"]
    

def feature(df):
    df["reserve_datediff"] = np.where(df["reserve_datetime_diff"] >= 7, 7, df["reserve_datetime_diff"])
    df1 = df[df["reserve_datediff"] >= 1]
    df2 = df[df["reserve_datediff"] >= 2]
    df3 = df[df["reserve_datediff"] >= 3]
    df4 = df[df["reserve_datediff"] >= 4]
    df5 = df[df["reserve_datediff"] >= 5]
    df6 = df[df["reserve_datediff"] >= 6]
    df7 = df[df["reserve_datediff"] >= 7]

    temp1 = df1.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg({np.mean, np.sum}).reset_index()
    temp1.columns = ["air_store_id", "visit_date", "mean1", "sum1"]
    temp2 = df2.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg({np.mean, np.sum}).reset_index()
    temp2.columns = ["air_store_id", "visit_date", "mean2", "sum2"]
    temp3 = df3.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg({np.mean, np.sum}).reset_index()
    temp3.columns = ["air_store_id", "visit_date", "mean3", "sum3"]
    temp4 = df4.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg({np.mean, np.sum}).reset_index()
    temp4.columns = ["air_store_id", "visit_date", "mean4", "sum4"]
    temp5 = df5.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg({np.mean, np.sum}).reset_index()
    temp5.columns = ["air_store_id", "visit_date", "mean5", "sum5"]
    temp6 = df6.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg({np.mean, np.sum}).reset_index()
    temp6.columns = ["air_store_id", "visit_date", "mean6", "sum6"]
    temp7 = df7.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg({np.mean, np.sum}).reset_index()
    temp7.columns = ["air_store_id", "visit_date", "mean7", "sum7"]

    df = pd.merge(temp1, temp2, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp3, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp4, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp5, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp6, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp7, on=["air_store_id", "visit_date"], how="left")
    df["datetime"] = pd.to_datetime(df['visit_date'])
    df["dow"] = df["datetime"].dt.dayofweek
    df["visit_date"] = df["visit_date"].apply(lambda x: x.strftime('%Y-%m-%d'))

    df["dow_reserve_sum"] = df.apply(get_dow_reserve_sum, axis=1)
    df["dow_reserve_mean"] = df.apply(get_dow_reserve_mean, axis=1)
    # print(df.head(10))

    return df


def calc_ewm(series, alpha, adjust=True):
    return series.ewm(alpha=alpha, adjust=adjust).mean()


def get_ewm_reserve(df):
    grouped = df.groupby(["air_store_num", "dows"])["sum1_air"]
    df["ewm_sum1_air"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(1))
    grouped = df.groupby(["air_store_num", "dows"])["sum1_hpg"]
    df["ewm_sum1_hpg"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(1))
    grouped = df.groupby(["air_store_num", "dows"])["mean1_air"]
    df["ewm_mean1_air"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(1))
    grouped = df.groupby(["air_store_num", "dows"])["mean1_hpg"]
    df["ewm_mean1_hpg"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(1))
    grouped = df.groupby(["air_store_num", "dows"])["total_reserve1"]
    df["ewm_sum1_tot"] = grouped.transform(lambda g: calc_ewm(g, 0.1).shift(1))
    return df


def get_total_info(df):
    df["total_reserve1"] = df["sum1_air"] + df["sum1_hpg"]
    df["total_reserve7"] = df["sum7_air"] + df["sum7_hpg"]
    return df


air_reserve_df = pd.read_csv('../input/air_reserve.csv')
hpg_reserve_df = pd.read_csv('../input/hpg_reserve.csv')
relation_df = pd.read_csv('../input/store_id_relation.csv')
train = pd.read_csv('../output/cws_train.csv')
predict = pd.read_csv('../output/cws_predict.csv')
print("Loaded data.")
hpg_reserve_df = pd.merge(hpg_reserve_df, relation_df, how='inner', on=['hpg_store_id'])
# air_reserve_df = air_reserve_df[air_reserve_df["air_store_id"] == "air_6b15edd1b4fbb96a"]
# hpg_reserve_df = hpg_reserve_df[hpg_reserve_df["air_store_id"] == "air_6b15edd1b4fbb96a"]
# print(air_reserve_df.head())

air_reserve_df = prepare(air_reserve_df)
hpg_reserve_df = prepare(hpg_reserve_df)
air_reserve_df = feature(air_reserve_df)
hpg_reserve_df = feature(hpg_reserve_df)
print("Done reserve engineer")

# print(air_reserve_df.describe())
# print(hpg_reserve_df.describe())
use_col = ["air_store_id", "visit_date", "dow_reserve_sum", "dow_reserve_mean",
           "sum1", "mean1", "sum7", "mean7"]
air_reserve_df = air_reserve_df[use_col]
hpg_reserve_df = hpg_reserve_df[use_col]
air_reserve_df.columns = ["air_store_id", "visit_date", "dow_reserve_sum_air", "dow_reserve_mean_air", 
                          "sum1_air", "mean1_air", "sum7_air", "mean7_air"]
hpg_reserve_df.columns = ["air_store_id", "visit_date", "dow_reserve_sum_hpg", "dow_reserve_mean_hpg",
                          "sum1_hpg", "mean1_hpg", "sum7_hpg", "mean7_hpg"]
air_reserve_df.fillna(0)
hpg_reserve_df.fillna(0)
# print(air_reserve_df.head())

joined = pd.concat([train, predict])
# joined = joined[joined["air_store_id"] == "air_6b15edd1b4fbb96a"]

# joined["visit_datetime"] = pd.to_datetime(joined["visit_date"])
# joined["visit_date"] = joined["visit_datetime"].dt.date
# print(joined.head())

print("Now merging...")
# Make sure that every open day with no reserve is filled with 0


joined = pd.merge(joined, air_reserve_df, on=["air_store_id", "visit_date"], how="left")
joined = pd.merge(joined, hpg_reserve_df, on=["air_store_id", "visit_date"], how="left")

joined = get_total_info(joined)
joined = get_ewm_reserve(joined)

# pd.set_option('display.width', 240)
# pd.set_option('display.max_columns', None)
print(joined.describe())


train = joined[joined["visit_date"] < "2017-04-23"]
predict = joined[joined["visit_date"] >= "2017-04-23"]

print("output to csv...")
train.to_csv('../output/cwsv_train.csv',float_format='%.6f', index=False)
predict.to_csv('../output/cwsv_predict.csv',float_format='%.6f', index=False)



