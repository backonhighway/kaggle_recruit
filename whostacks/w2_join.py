import pandas as pd
import custom_metrics
import numpy as np


def get_joined_model(filenames):
    model_list = []
    for filename in filenames:
        model = pd.read_csv(filename)
        model_list.append(model)
    return pd.concat(model_list)


def get_filenames(prefix, suffix):
    period_list = ["0416_0422", "0409_0415", "0402_0408", "0312_0319"]
    name_list = []
    for period in period_list:
        name_list.append(prefix + period + suffix)
    return name_list


# load data
prefix_w2 = "../output/p1_w2_"
normal_filenames = get_filenames(prefix_w2, ".csv")
w2_normal = get_joined_model(normal_filenames)
good_filenames = get_filenames(prefix_w2, "_good.csv")
w2_good = get_joined_model(good_filenames)
short_filenames = get_filenames(prefix_w2, "_short.csv")
w2_short = get_joined_model(short_filenames)
print("Loaded data")


print(w2_normal["actual"].count())
print(w2_good["actual"].count())
print(w2_short["actual"].count())
print("-"*40)

df = pd.merge(w2_normal, w2_short, on=["air_store_id", "visit_date"], suffixes=["_normal", "_short"])

normal_score = custom_metrics.rmse(df["actual_log_normal"], df["prediction_normal"])
short_score = custom_metrics.rmse(df["actual_log_normal"], df["prediction_short"])
print("normal=", normal_score)
print("short=", short_score)


df = pd.merge(df, w2_good, on=["air_store_id", "visit_date"], how="left", suffixes=["", "_good"])
df["with_good"] = np.where(df["prediction"].isnull(), df["prediction_normal"], df["prediction"])
with_good_score = custom_metrics.rmse(df["actual_log_normal"], df["with_good"])
print("normal+good=", with_good_score)