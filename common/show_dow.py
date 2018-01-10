import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# load data
train = pd.read_csv('../output/fed_train.csv')
predict = pd.read_csv('../output/fed_predict.csv')
hol_df = pd.read_csv("../input/date_info.csv")

print("loaded data.")

train["visit_datetime"] = pd.to_datetime(train["visit_date"])
train["next_datetime"] = train["visit_datetime"].dt.date + pd.DateOffset(days=1)
train["next_date"] = train["next_datetime"].dt.date
hol_df = hol_df.rename(columns={"calendar_date": "next_date", "holiday_flg": "next_holiday_flg"})
train["next_date_str"] = train["next_date"].astype(str)
hol_df["next_date_str"] = hol_df["next_date"].astype(str)
train = pd.merge(train, hol_df, how="left", on="next_date_str")
train["next_dow"] = train["next_datetime"].dt.dayofweek
train["next_dowh"] = np.where((train["next_holiday_flg"] == 1) & (train["next_dow"] < 5), 7, train["next_dow"])
train["next_is_hol"] = np.where(train["next_dowh"] >= 5, 1, 0)


# train = train[train["month"] > 1]

sat = train[train["dowh"] == 5]
sun = train[train["dowh"] == 6]
hol = train[train["dowh"] == 7]
print("-"*40)
nh = train[(train["next_is_hol"] == 1) & (train["holiday_flg"] == 1)]
nnh = train[(train["next_is_hol"] == 0) & (train["holiday_flg"] == 1)]
print(sat["visitors"].describe())
print(sun["visitors"].describe())
print(nh["visitors"].describe())
print(nnh["visitors"].describe())
exit(0)

print(sat.describe()["visitors"])
print(sun.describe()["visitors"])
print(hol.describe()["visitors"])
print(nh.describe()["visitors"])

sns.distplot(sat["visitors"], hist=False, color="blue")
sns.distplot(sun["visitors"], hist=False, color="red")
sns.distplot(hol["visitors"], hist=False, color="green")
sns.distplot(nh["visitors"], hist=False, color="yellow")
sns.plt.show()