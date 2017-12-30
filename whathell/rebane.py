import pandas as pd
from sklearn import preprocessing

import custom_lgb

le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()

data = {
    'train': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
#    'ar': pd.read_csv('../input/air_reserve.csv'),
#    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'predict': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

data['train']['visit_date_str'] = pd.to_datetime(data['train']['visit_date'])
data['train']['dow'] = data['train']['visit_date_str'].dt.dayofweek
data['train']['visit_date'] = data['train']['visit_date_str'].dt.date
#le.fit(df_as['air_genre_name'])
#df_as['air_genre_name'] = le.fit_transform(df_as['air_genre_name'])

#le.fit(df_as['air_area_name'])
#df_as['air_area_name'] = le.fit_transform(df_as['air_area_name'])
le.fit(data['train']['air_store_id'])
data['train']['air_store_num'] = le.transform(data['train']['air_store_id'])

data['predict']['visit_date_str'] = data['predict']['id'].map(lambda x: str(x).split('_')[2])
data['predict']['air_store_id'] = data['predict']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['predict']['visit_date'] = pd.to_datetime(data['predict']['visit_date_str'])
data['predict']['dow'] = data['predict']['visit_date'].dt.dayofweek
data['predict']['visit_date'] = data['predict']['visit_date'].dt.date
data['predict']['air_store_num'] = le.transform(data['predict']['air_store_id'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['train'], data['hol'], how='left', on=['visit_date'])
predict = pd.merge(data['predict'], data['hol'], how='left', on=['visit_date'])

# print(train.head())
# print(predict.head())
# train.to_csv('../output/cleaned_train.csv',float_format='%.6f', index=False)


# make input
train_input = train[ (train['visit_date_str'] >= '2016-01-01') & (train['visit_date_str'] <= '2016-12-01') ].reset_index(drop=True)
test_input = train[ (train['visit_date_str'] >= '2017-03-01') & (train['visit_date_str'] <= '2017-04-01') ].reset_index(drop=True)

col = ['air_store_num', 'visitors', 'dow', 'holiday_flg']
train_input = train_input[col]
test_input = test_input[col]
x_pred = predict[col].drop('visitors', axis=1)

print(train_input.head())
print(test_input.head())
print(x_pred.head())
print('-' * 30)

y_pred = custom_lgb.do_predict(train_input, test_input, x_pred)
y_pred[y_pred < 0] = 0

submission = pd.DataFrame({
        "id": data['predict']['id'],
        "visitors": y_pred
    })
print(submission.describe())
submission.to_csv('../output/submission.csv',float_format='%.6f', index=False)


