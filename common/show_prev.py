import pandas as pd
import pocket_periods


train = pd.read_csv('../output/cwsvr_train.csv')
prev_stores = pd.read_csv("../output/prev_year_store.csv")
train = train[train["air_store_num"].isin(prev_stores["air_store_num"]) == False]

print(train.tail())

print(train.isnull().sum())