import pandas as pd
import numpy as np
import lolpy


SHIFT_WEEKS = 2


def prepare(df):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
    df['visit_date'] = df['visit_datetime'].dt.date
    df["dow"] = df["visit_datetime"].dt.dayofweek
    df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
    df['reserve_date'] = df['reserve_datetime'].dt.date
    df['reserve_datetime_diff'] = df.apply(lambda r: (r['visit_date'] - r['reserve_date']).days, axis=1)
    return df
    

def feature(df):
    temp0 = df.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp0.columns = ["air_store_id", "visit_date", "r_sum0"]

    df7 = df[df["reserve_datetime_diff"] >= 7]
    temp7 = df7.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp7.columns = ["air_store_id", "visit_date", "r_sum7"]
    df8 = df[df["reserve_datetime_diff"] >= 8]
    temp8 = df8.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp8.columns = ["air_store_id", "visit_date", "r_sum8"] 
    df9 = df[df["reserve_datetime_diff"] >= 9]
    temp9 = df9.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp9.columns = ["air_store_id", "visit_date", "r_sum9"] 
    df10 = df[df["reserve_datetime_diff"] >= 10]
    temp10 = df10.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp10.columns = ["air_store_id", "visit_date", "r_sum10"] 
    df11 = df[df["reserve_datetime_diff"] >= 11]
    temp11 = df11.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp11.columns = ["air_store_id", "visit_date", "r_sum11"] 
    df12 = df[df["reserve_datetime_diff"] >= 12]
    temp12 = df12.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp12.columns = ["air_store_id", "visit_date", "r_sum12"] 
    df13 = df[df["reserve_datetime_diff"] >= 13]
    temp13 = df13.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp13.columns = ["air_store_id", "visit_date", "r_sum13"]
    df14 = df[df["reserve_datetime_diff"] >= 14]
    temp14 = df14.groupby(["air_store_id", "visit_date"])["reserve_visitors"].agg(np.sum).reset_index()
    temp14.columns = ["air_store_id", "visit_date", "r_sum14"]

    df = pd.merge(temp0, temp7, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp8, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp9, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp10, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp11, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp12, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp13, on=["air_store_id", "visit_date"], how="left")
    df = pd.merge(df, temp14, on=["air_store_id", "visit_date"], how="left")

    df["datetime"] = pd.to_datetime(df['visit_date'])
    df["dow"] = df["datetime"].dt.dayofweek
    df["visit_date"] = df["visit_date"].apply(lambda x: x.strftime('%Y-%m-%d'))
    # print(df.head(20))
    df.fillna(0)

    shift_col(df, "r_sum0")
    # print("-"*40)
    # print(df.head(20))
    # exit(0)

    # todo add ewm, ratio
    return df


def shift_col(df, col_name):
    next_col_name = col_name + "_shifted"
    shift_period = SHIFT_WEEKS * 7
    df[next_col_name] = df.groupby(["air_store_id"])[col_name].transform(lambda g: g.shift(shift_period))

    next_col_name = "dow_" + col_name + "_shifted"
    shift_period = SHIFT_WEEKS
    df[next_col_name] = df.groupby(["air_store_id", "dow"])[col_name].transform(lambda g: g.shift(shift_period))


def rename_col(df, prefix):
    base_col = ["air_store_id", "visit_date"]
    res_col = [
        "r_sum0", "r_sum7", "r_sum8", "r_sum9", "r_sum10", "r_sum11", "r_sum12", "r_sum13", "r_sum14",
        "r_sum0_shifted", "dow_r_sum0_shifted"
    ]
    now_col = base_col + res_col
    print(now_col)
    df = df[now_col]

    next_res_col = [prefix + s for s in res_col]
    next_col = base_col + next_res_col
    print(next_col)
    df.columns = next_col

    return df


file_prefix = "../output/outlier_w2_"
air_reserve_df = pd.read_csv('../input/air_reserve.csv')
hpg_reserve_df = pd.read_csv('../input/hpg_reserve.csv')
relation_df = pd.read_csv('../input/store_id_relation.csv')
train = pd.read_csv(file_prefix + "cwrrs_train.csv")
predict = pd.read_csv(file_prefix + "cwrrs_predict.csv")
print("Loaded data.")
hpg_reserve_df = pd.merge(hpg_reserve_df, relation_df, how='inner', on=['hpg_store_id'])
#air_reserve_df = air_reserve_df[air_reserve_df["air_store_id"] == "air_6b15edd1b4fbb96a"]
#hpg_reserve_df = hpg_reserve_df[hpg_reserve_df["air_store_id"] == "air_6b15edd1b4fbb96a"]
# print(air_reserve_df.head())

air_reserve_df = prepare(air_reserve_df)
hpg_reserve_df = prepare(hpg_reserve_df)
air_reserve_df = feature(air_reserve_df)
hpg_reserve_df = feature(hpg_reserve_df)
print("Done reserve engineer")

# print(air_reserve_df.describe())
# print(hpg_reserve_df.describe())
air_reserve_df = rename_col(air_reserve_df, "air_")
hpg_reserve_df = rename_col(hpg_reserve_df, "hpg_")
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



# pd.set_option('display.width', 240)
# pd.set_option('display.max_columns', None)
# print(joined.describe())


train = joined[joined["visit_date"] < "2017-04-23"]
predict = joined[joined["visit_date"] >= "2017-04-23"]

print("output to csv...")
train.to_csv(file_prefix + 'cwrrsr_train.csv',float_format='%.6f', index=False)
predict.to_csv(file_prefix + 'cwrrsr_predict.csv',float_format='%.6f', index=False)



