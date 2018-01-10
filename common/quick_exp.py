import pandas as pd

train =  pd.read_csv('../input/air_store_info.csv')


train["prefecture"] = train["air_area_name"].map(lambda x: str(x).split()[0])
train["city"] = train["air_area_name"].map(lambda x: str(x).split()[1])
print(train.head())