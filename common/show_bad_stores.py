import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('../output/cwsvr_train.csv')
use_col = ["air_store_num", "dow", "visitors", "visit_date", "dows"]
train = train[use_col]
train = train[train["visit_date"] >= "2017-01-05"]

bad_stores = pd.read_csv("../output/bad_stores.csv")
# print(bad_stores.head())

train = train[train["air_store_num"] == 511]
train = train[["visit_date", "visitors"]]
# print(train.head())
# train["visit_datetime"] = pd.to_datetime(train["visit_date"], format='%Y-%m-%d')
# train["visit_datetime"] = train["visit_datetime"].dt.date

train.reset_index(inplace=True)

train.plot(x="visit_date", y="visitors")
plt.show()