import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('../output/cleaned_wr_train.csv')
predict = pd.read_csv('../output/cleaned_wr_predict.csv')

col = ["air_store_num", "visit_date", "visitors", "precipitation", "avg_temperature"]
train = train[col]
print(train.describe())
train["is_rainy"] = np.where(train["precipitation"] > 0, 1, 0)
train["is_very_rainy"] = np.where(train["precipitation"] > 4, 1, 0)

no_per = train[train["precipitation"].isnull()]
print(no_per["visitors"].describe())
no_temp = train[train["avg_temperature"].isnull()]
print(no_temp["visitors"].describe())

print(train.groupby("is_rainy")["visitors"].describe())
print(train.groupby("is_very_rainy")["visitors"].describe())

sns.violinplot(x="precipitation", y="visitors", data=train)
plt.show()