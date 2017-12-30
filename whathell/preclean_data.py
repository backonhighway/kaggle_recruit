import pandas as pd
from sklearn import preprocessing


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
data['train']['dow'] = data['train']['visit_date_str'].dt.dayofweek
data['train']['visit_date'] = data['train']['visit_date_str'].dt.date
data['train']['air_store_num'] = le.fit_transform(data['train']['air_store_id'])

data['predict']['visit_date_str'] = data['predict']['id'].map(lambda x: str(x).split('_')[2])
data['predict']['air_store_id'] = data['predict']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['predict']['visit_date'] = pd.to_datetime(data['predict']['visit_date_str'])
data['predict']['dow'] = data['predict']['visit_date'].dt.dayofweek
data['predict']['visit_date'] = data['predict']['visit_date'].dt.date
data['predict']['air_store_num'] = le.fit_transform(data['predict']['air_store_id'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

data['as']['air_genre_num'] = le.fit_transform(data['as']['air_genre_name'])
data['as']['air_area_num'] = le.fit_transform(data['as']['air_area_name'])

train = pd.merge(data['train'], data['hol'], how='left', on=['visit_date'])
predict = pd.merge(data['predict'], data['hol'], how='left', on=['visit_date'])
train = pd.merge(train, data['as'], how='left', on=['air_store_id'])
predict = pd.merge(predict, data['as'], how='left', on=['air_store_id'])

train_col = ['air_store_num',  'visitors', 'visit_date', 'visit_date_str', 'dow', 'holiday_flg',
             'air_genre_num', 'air_area_num']
predict_col = ['id', 'air_store_num',  'visitors', 'visit_date', 'visit_date_str', 'dow', 'holiday_flg',
             'air_genre_num', 'air_area_num']

# print(train.head())
# print(predict.head())

train = train[train_col]
predict = predict[predict_col]

train.to_csv('../output/cleaned_train.csv', float_format='%.6f', index=False)
predict.to_csv('../output/cleaned_predict.csv', float_format='%.6f', index=False)