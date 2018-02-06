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
out_filenames = get_filenames(prefix_w2, "_out.csv")
w2_out = get_joined_model(out_filenames)
nores_filenames = get_filenames(prefix_w2, "_no_res.csv")
w2_nores = get_joined_model(nores_filenames)
print("Loaded data")

# print(w2_normal["prediction"].count())
# print(w2_good["prediction"].count())
# print(w2_short["prediction"].count())
# print(w2_out["prediction"].count())
# print(w2_nores["prediction"].count())
print("-"*40)
normal_score = custom_metrics.rmse(w2_normal["actual_log"], w2_normal["prediction"])
short_score = custom_metrics.rmse(w2_short["actual_log"], w2_short["prediction"])
out_score = custom_metrics.rmse(w2_out["actual_log"], w2_out["prediction"])
nores_score = custom_metrics.rmse(w2_nores["actual_log"], w2_nores["prediction"])
print("normal=", normal_score)
print("short=", short_score)
print("out=", out_score)
print("nores=", nores_score)
print("-"*40)


df = pd.merge(w2_normal, w2_short, how="left", on=["air_store_id", "visit_date"], suffixes=["_normal", "_short"])
df = pd.merge(df, w2_good, on=["air_store_id", "visit_date"], how="left", suffixes=["", "_good"])
df = pd.merge(df, w2_out, on=["air_store_id", "visit_date"], how="left", suffixes=["", "_out"])
df = pd.merge(df, w2_nores, on=["air_store_id", "visit_date"], how="left", suffixes=["", "_nores"])

df["with_good"] = np.where(df["prediction"].isnull(), df["prediction_normal"], df["prediction"])
with_good_score = custom_metrics.rmse(df["actual_log_normal"], df["with_good"])
print("normal+good=", with_good_score)

df["avg_nor_out"] = df["prediction_normal"] * 0.7 + df["prediction_out"] * 0.3
avg_nor_out_score = custom_metrics.rmse(df["actual_log_normal"], df["avg_nor_out"])
print("normal+out=", avg_nor_out_score)

df["avg_nor_out_good"] = df["with_good"] * 0.6 + df["prediction_out"] * 0.4
avg_nor_out_score = custom_metrics.rmse(df["actual_log_normal"], df["avg_nor_out_good"])
print("normal+out+good=", avg_nor_out_score)

df["avg_nores"] = df["avg_nor_out_good"] * 0.9 + df["prediction_nores"] * 0.1
avg_nores = custom_metrics.rmse(df["actual_log_normal"], df["avg_nores"])
print("avg_nores=", avg_nores)


print("-"*40)
correlation_score = df["prediction_normal"].corr(df["with_good"])
print(correlation_score)
correlation_score = df["prediction_normal"].corr(df["prediction_short"])
print(correlation_score)
correlation_score = df["prediction_normal"].corr(df["prediction_out"])
print(correlation_score)
correlation_score = df["prediction_short"].corr(df["prediction_out"])
print(correlation_score)
correlation_score = df["actual_log_normal"].corr(df["avg_nor_out_good"])
print(correlation_score)



