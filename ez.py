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
df["prev_month"] = df["date"] - offsets.MonthBegin(2)

temp = df.groupby(["air_store_id", "month"]).agg({"visitors": ["min", "max"]})
print(temp.head())


exit(0)

df["moving_average"] = df["visitors"].rolling(3, min_periods=0, ).mean()
df["moving_average_2"] = df.groupby("air_store_id")["visitors"].rolling(3, min_periods=0).mean().reset_index(0, drop=True)
print(df)
df["moving_average_2"] = df.groupby("air_store_id")["moving_average_2"].transform(lambda x: x.shift(1))
print(df)
