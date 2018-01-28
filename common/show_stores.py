import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('../output/all_c_train.csv')
use_col = ["air_store_num", "dow", "visitors", "visit_date", "dows"]
train = train[use_col]

test = pd.read_csv("../output/all_c_predict.csv")
test = test[use_col]

train_stores = pd.DataFrame(train["air_store_num"].unique())
train_stores.columns = ["store_num"]
print(train_stores.describe())

test_stores = pd.DataFrame(test["air_store_num"].unique())
test_stores.columns = ["store_num"]
print(test_stores.describe())

#train_stores = train["air_store_num"].unique()
wtf = list(test["air_store_num"].unique())
diff = train_stores[~train_stores["store_num"].isin(wtf)]
print(diff)