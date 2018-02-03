import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# load data
train = pd.read_csv('../output/all_c_train.csv')

print("loaded data.")


print(train["visitors"].describe())
out17 = train[train["is_outlier02"]]

print(out17["month"].value_counts())
print(out17["year"].value_counts())
print(out17["air_store_num"].value_counts())


