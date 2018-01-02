import pandas as pd
from sklearn import preprocessing
import numpy as np
import pandas.tseries.offsets as offsets

le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()

data = {
    'train': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    #    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    #    'ar': pd.read_csv('../input/air_reserve.csv'),
    #    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    #    'id': pd.read_csv('../input/store_id_relation.csv'),
    'predict': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

data['train']['visit_date_str'] = pd.to_datetime(data['train']['visit_date'])
data['train']['visit_date'] = data['train']['visit_date_str'].dt.date

data['predict']['visit_date_str'] = data['predict']['id'].map(lambda x: str(x).split('_')[2])
data['predict']['air_store_id'] = data['predict']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['predict']['visit_date'] = pd.to_datetime(data['predict']['visit_date_str'])
data['predict']['visit_date'] = data['predict']['visit_date'].dt.date

data['hol']['visit_datetime'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_datetime'].dt.date
data['hol']["year"] = data['hol']["visit_datetime"].dt.year
data['hol']['dow'] = data['hol']['visit_datetime'].dt.dayofweek
data['hol']["week"] = data['hol']["visit_datetime"].dt.week
data['hol']["next_week"] = data['hol']["visit_datetime"] + offsets.Week(1)
data['hol']["prev_week"] = data['hol']["visit_datetime"] - offsets.Week(1)
data['hol']["next_week"] = data['hol']["next_week"].dt.week
data['hol']["prev_week"] = data['hol']["prev_week"].dt.week
data['hol']['dowh_flg'] = np.where((data['hol']["holiday_flg"] == 1) & (data['hol']["dow"] < 5), 1, 0)
grouped = data['hol'].groupby(["year", "week"])["dowh_flg"].sum().reset_index()  # Todo: perhaps not perfect
grouped.columns = ["year", "week", "week_hols"]
data['hol'] = pd.merge(data['hol'], grouped, how='left', on=["year", "week"])
grouped.columns = ["year", "next_week", "next_week_hols"]
data['hol'] = pd.merge(data['hol'], grouped, how='left', on=["year", "next_week"])
grouped.columns = ["year", "prev_week", "prev_week_hols"]
data['hol'] = pd.merge(data['hol'], grouped, how='left', on=["year", "prev_week"])

data['as']['air_store_num'] = le.fit_transform(data['as']['air_store_id'])
data['as']['air_genre_num'] = le.fit_transform(data['as']['air_genre_name'])
data['as']['air_area_num'] = le.fit_transform(data['as']['air_area_name'])

train = pd.merge(data['train'], data['hol'], how='left', on=['visit_date'])
predict = pd.merge(data['predict'], data['hol'], how='left', on=['visit_date'])
train = pd.merge(train, data['as'], how='left', on=['air_store_id'])
predict = pd.merge(predict, data['as'], how='left', on=['air_store_id'])

train_col = ['air_store_num', 'visitors', 'visit_date', 'dow', 'holiday_flg',
             'week_hols', 'next_week_hols', 'prev_week_hols',
             'air_genre_num', 'air_area_num']
predict_col = ['id', 'air_store_num', 'visitors', 'visit_date', 'dow', 'holiday_flg',
               'week_hols', 'next_week_hols', 'prev_week_hols'
                'air_genre_num', 'air_area_num', ]

# print(train.head())
# print(predict.head())

train = train[train_col]
predict = predict[predict_col]

train.to_csv('../output/cleaned_train.csv', float_format='%.6f', index=False)
predict.to_csv('../output/cleaned_predict.csv', float_format='%.6f', index=False)
