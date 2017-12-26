import pandas as pd
import time

start_time = time.time()
df = pd.read_csv("./input/pd_test.csv")
read_time = time.time() - start_time
print(read_time)

df["date"] = pd.to_datetime(df["visit_date"])
df["dow"] = df["date"].dt.dayofweek

split = df[df["date"] <= "20160114"]

df["moving_average"] = df["visitors"].rolling(3).mean().shift(1)
df["moving_average_2"] = df.groupby("air_store_id")["visitors"].rolling(3).mean().reset_index(0, drop=True)
print(df)
print(split.head())