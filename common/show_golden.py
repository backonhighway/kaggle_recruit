import pandas as pd

train = pd.read_csv('../output/w2_cwrrsr_train.csv')
print(train["visitors"].describe())
print("-" * 40)

golden = train[(train["visit_date"] >= "2016-04-28") & (train["visit_date"] <= "2016-05-09")]
print(golden["visitors"].describe())
print(golden.groupby("visit_date")["visitors"].describe())
print(golden.groupby("visit_date")["air_r_sum7l"].sum())
print(golden.groupby("visit_date")["hpg_r_sum7l"].sum())
print(golden.groupby("visit_date")["air_r_sum7"].sum())
print(golden.groupby("visit_date")["hpg_r_sum7"].sum())

near = train[train["visit_date"] >= "2017-04-02"]
#print(near["visitors"].describe())
print("-"*40)

predict = pd.read_csv('../output/w2_cwrrsr_predict.csv')
goldenp = predict[(predict["visit_date"] >= "2017-04-28") & (predict["visit_date"] <= "2017-05-09")]
print(goldenp.groupby("visit_date")["air_r_sum7l"].sum())
print(goldenp.groupby("visit_date")["hpg_r_sum7l"].sum())
print(goldenp.groupby("visit_date")["air_r_sum7"].sum())
print(goldenp.groupby("visit_date")["hpg_r_sum7"].sum())
