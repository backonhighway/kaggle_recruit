import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# load data
train = pd.read_csv('../output/fed_train.csv')
predict = pd.read_csv('../output/fed_predict.csv')
hol_df = pd.read_csv("../input/date_info.csv")

print("loaded data.")

out = train[train["visitors"] >= 70]

print(train["visitors"].describe())
print(out["visitors"].describe())
print(out.groupby("dowh")["visitors"].describe())

print("-"*40)

print(train.groupby("air_store_num")["visitors"].describe())
print(out.groupby("air_store_num")["visitors"].describe())


