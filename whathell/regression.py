from sklearn import linear_model
import pandas as pd
import pandas.tseries.offsets as offsets
import seaborn as sns
import matplotlib.pyplot as plt


def regress(df, x_pred):
    X = pd.DataFrame(df["day_delta"])
    y = pd.DataFrame(df["visitors"])

    reg = linear_model.Ridge(alpha=.5)
    return reg.fit(X, y).predict(x_pred)

df = pd.DataFrame({})







# load data
train = pd.read_csv('../output/cleaned_train.csv')
predict = pd.read_csv('../output/cleaned_predict.csv')

X = pd.DataFrame(train["day_delta"])
y = pd.DataFrame(train["visitors"])

reg = linear_model.Ridge(alpha=.5)
reg.fit(X, y)