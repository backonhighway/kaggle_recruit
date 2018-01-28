import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def prepare(df):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
    df['visit_date'] = df['visit_datetime'].dt.date
    df["dow"] = df["visit_datetime"].dt.dayofweek
    df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
    df['reserve_date'] = df['reserve_datetime'].dt.date
    df['reserve_datetime_diff'] = df.apply(lambda r: (r['visit_date'] - r['reserve_date']).days, axis=1)
    df = df[df["reserve_datetime_diff"] >= 7]
    return df


air_reserve_df = pd.read_csv('../input/air_reserve.csv')
hpg_reserve_df = pd.read_csv('../input/hpg_reserve.csv')
relation_df = pd.read_csv('../input/store_id_relation.csv')

train = pd.read_csv('../output/cws_train.csv')
predict = pd.read_csv('../output/cws_predict.csv')

print("Loaded data.")
hpg_reserve_df = pd.merge(hpg_reserve_df, relation_df, how='inner', on=['hpg_store_id'])

air_reserve_df = prepare(air_reserve_df)
hpg_reserve_df = prepare(hpg_reserve_df)

print(air_reserve_df["reserve_datetime_diff"].describe(include="all"))
print(hpg_reserve_df["reserve_datetime_diff"].describe(include="all"))
exit(0)
print("showing graph...")
sns.countplot(x="reserve_datetime_diff", data=air_reserve_df)
plt.show()

all_reserve = pd.concat([air_reserve_df, hpg_reserve_df])
print(all_reserve["reserve_datetime_diff"].describe(include="all"))