import pandas as pd


train = pd.read_csv('./input/pd_test.csv')


train["datetime"] = pd.to_datetime(train["visit_date"])
train["week"] = train["datetime"].dt.week
print(train.tail())
