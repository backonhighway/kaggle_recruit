import pandas as pd
import pocket_periods


train = pd.read_csv('../output/cwsvr_train.csv')

df_list = pocket_periods.get_df_list(train, pocket_periods.get_prev_year_period_list())
for df in df_list:
    print(df["visitors"].describe())

print("-" * 40)

df_list = pocket_periods.get_six_week_df_list(train)
for df in df_list:
    print(df["visitors"].describe())