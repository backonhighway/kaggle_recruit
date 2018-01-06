import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas.tseries.offsets as offsets


data = {
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
print("read")
for df in ['ar', 'hr']:
    data[df]["visit_datetime"] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]["visit_date"] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_date'] = data[df]['reserve_datetime'].dt.date
    print("not ok?")
    data[df]["reserve_diff"] = data[df].apply(
        lambda r:(r['visit_date'] - r['reserve_date']).days, axis=1)
    print("uok?")

print(data["hr"].describe())
grouped = data["hr"].groupby("reserve_diff")

sns.countplot(x="reserve_diff",  data=data["ar"])

plt.show()