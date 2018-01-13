import pandas as pd
import time
import pandas.tseries.offsets as offsets
import numpy as np

start_time = time.time()
df = pd.read_csv("./input/pd_test.csv")
read_time = time.time() - start_time
print(read_time)

df["date"] = pd.to_datetime(df["visit_date"])
df["month"] = df["date"].dt.month

# temp = df.groupby(["air_store_id", "month"])["visitors"].transform(np.mean)
# print(temp)

# df["dow"] = df["date"].dt.dayofweek
df["prev_date"] = df["date"] - offsets.Day(40)
df["prev_month"] = df["date"] - pd.DateOffset(months=1)
print(df)

temp = df.groupby(["air_store_id", "month"]).agg({"visitors": ["min", "max"]})
print(temp.head())


df["moving_average"] = df["visitors"].rolling(3, min_periods=0, ).mean()
df["moving_average_2"] = df.groupby("air_store_id")["visitors"].rolling(3, min_periods=0).mean().reset_index(0, drop=True)
print(df)
df["moving_average_2"] = df.groupby("air_store_id")["moving_average_2"].transform(lambda x: x.shift(1))
print(df)
df["moving_average_3"] = df.groupby("air_store_id")["visitors"].rolling(3, min_period=0).mean().reset_index(0, drop=True).transform(lambda x: x.shift(1))
print(df)

f = lambda x: x.rolling(3, min_periods=0).mean().shift(1)
df["trans_test"] = df.groupby(["air_store_id","month"])["visitors"].transform(f)

df["nan_test"] = df["moving_average_2"].rolling(3, min_periods=0, ).mean()
print(df)