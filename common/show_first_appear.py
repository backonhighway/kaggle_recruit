import pandas as pd

train = pd.read_csv('../output/cwsvr_train.csv')
use_col = ["air_store_num", "dow", "visitors", "visit_date", "dows"]
train = train[use_col]

grouped = train.groupby("air_store_num")["visit_date"].apply(min).reset_index()
#print(grouped["visit_date"].value_counts())
has_prev_year_data = grouped[grouped["visit_date"] <= "2016-03-04"]
print(has_prev_year_data.describe())
has_prev_year_data = has_prev_year_data.reset_index(drop=True)
has_prev_year_data.to_csv('../output/prev_year_store.csv', float_format='%.6f', index=False)

train["first_appear"] = train.groupby("air_store_num")["visit_date"].transform(min)
#print(train.head())
has_prev_year_data = train[train["first_appear"] <= "2017-04-23"]
#print(has_prev_year_data.describe())