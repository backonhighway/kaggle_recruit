import pandas as pd
import numpy as np


train = pd.read_csv('../output/cwsvr_train.csv')
use_col = ["air_store_num", "dow", "visitors", "visit_date", "dows"]
train = train[use_col]

grouped = train.groupby("air_store_num")["visitors"].agg(np.std)
print(grouped.describe())
woot = grouped.nlargest(30)
print(woot.head())
print(woot.tail())
store_list = woot["air_store_num"]
print(store_list)
