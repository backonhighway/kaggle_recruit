import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# load data
train = pd.read_csv('../output/fed_train.csv')
predict = pd.read_csv('../output/fed_predict.csv')
hol_df = pd.read_csv("../input/date_info.csv")

print("loaded data.")

train = train[train["visitors"] <= 100]
sat = train[train["dow"] == 5]
sun = train[train["dow"] == 6]
hol = train[train["holiday_flg"] == 1]
print("-"*40)

# print(sat.describe()["visitors"])
# print(sun.describe()["visitors"])
# print("-"*30)
# print(train.groupby("dow")["visitors"].describe())
print(train.groupby("dowh")["visitors"].describe())
print("-"*40)
print(hol.groupby("dow")["visitors"].describe())
print("-"*40)
print(hol.groupby(["dow","visit_date"])["visitors"].describe())

sns.distplot(sat["visitors"], hist=False, color="blue")
sns.distplot(sun["visitors"], hist=False, color="red")
sns.distplot(hol["visitors"], hist=False, color="green")
sns.plt.show()