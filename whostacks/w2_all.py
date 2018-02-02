import pandas as pd
import custom_metrics
import numpy as np

# load data
all_model = pd.read_csv('../output/p1_all_0416_0422.csv')
w2_model = pd.read_csv('../output/p1_w2_0416_0422.csv')
w2_good = pd.read_csv('../output/p1_w2_0416_0422_good_stores.csv')
w2_short = pd.read_csv('../output/p1_w2_0416_0422_short_train.csv')
print("Loaded data")

print(all_model["actual"].count())
print(w2_model["actual"].count())
print(w2_good["actual"].count())
print(w2_short["actual"].count())
print("-"*40)

df = pd.merge(all_model, w2_model, on=["air_store_id", "visit_date"], suffixes=["_all", "_w2"])
# print(df.describe())

all_score = custom_metrics.rmse(df["actual_log_all"], df["prediction_all"])
w2_score = custom_metrics.rmse(df["actual_log_all"], df["prediction_w2"])
print("all=", all_score)
print("w2=", w2_score)

df["averaged"] = df["prediction_all"] * 0.1 + df["prediction_w2"] * 0.9
avg_score = custom_metrics.rmse(df["actual_log_all"], df["averaged"])
print("all+w2=", avg_score)

df = pd.merge(df, w2_good, on=["air_store_id", "visit_date"], how="left", suffixes=["", "_good"])
df["with_good"] = np.where(df["prediction"].isnull(), df["prediction_w2"], df["prediction"])
with_good_score = custom_metrics.rmse(df["actual_log_all"], df["with_good"])
print("w2+good=", with_good_score)

df = pd.merge(df, w2_short, on=["air_store_id", "visit_date"], how="left", suffixes=["", "_short"])
df["avg_with_short"] = df["prediction_short"] * 0.1 + df["prediction_w2"] * 0.9
with_good_score = custom_metrics.rmse(df["actual_log_all"], df["avg_with_short"])
print("short+w2=", with_good_score)

df["good_with_short"] = df["prediction_short"] * 0.1 + df["with_good"] * 0.9
short_good_score = custom_metrics.rmse(df["actual_log_all"], df["good_with_short"])
print("short+good=", short_good_score)



print("-"*40)
correlation_score = df["prediction_all"].corr(df["prediction_w2"])
print(correlation_score)
correlation_score = df["prediction_w2"].corr(df["with_good"])
print(correlation_score)
correlation_score = df["prediction_w2"].corr(df["prediction_short"])
print(correlation_score)
correlation_score = df["with_good"].corr(df["prediction_short"])
print(correlation_score)


