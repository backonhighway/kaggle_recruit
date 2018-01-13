import pandas as pd
from sklearn import preprocessing
import numpy as np
import pandas.tseries.offsets as offsets

le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()


data = {
    'train': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'predict': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}


data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])


for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                       axis=1)
    data[df] = data[df][data[df]["reserve_datetime_diff"] > 0]
    tmp1 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
    tmp2 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])

data['train']['visit_date_str'] = pd.to_datetime(data['train']['visit_date'])
data['train']['visit_date'] = data['train']['visit_date_str'].dt.date

data['predict']['visit_date_str'] = data['predict']['id'].map(lambda x: str(x).split('_')[2])
data['predict']['air_store_id'] = data['predict']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['predict']['visit_date'] = pd.to_datetime(data['predict']['visit_date_str'])
data['predict']['visit_date'] = data['predict']['visit_date'].dt.date

data['hol']['visit_datetime'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_datetime'].dt.date
data['hol']["year"] = data['hol']["visit_datetime"].dt.year
data['hol']["month"] = data['hol']["visit_datetime"].dt.month
data['hol']["prev_month"] = data['hol']["visit_datetime"] - pd.DateOffset(months=1)
data['hol']["prev_month_y"] = data['hol']["prev_month"].dt.year
data['hol']["prev_month_m"] = data['hol']["prev_month"].dt.month
data['hol']['dow'] = data['hol']['visit_datetime'].dt.dayofweek
data['hol']["week"] = data['hol']["visit_datetime"].dt.week
data['hol']["next_week"] = data['hol']["visit_datetime"] + offsets.Week(1)
data['hol']["prev_week"] = data['hol']["visit_datetime"] - offsets.Week(1)
data['hol']["next_week"] = data['hol']["next_week"].dt.week
data['hol']["prev_week"] = data['hol']["prev_week"].dt.week
data['hol']['dowh_flg'] = np.where((data['hol']["holiday_flg"] == 1) & (data['hol']["dow"] < 5), 1, 0)
grouped = data['hol'].groupby(["year", "week"])["dowh_flg"].sum().reset_index()
grouped.columns = ["year", "week", "week_hols"]
data['hol'] = pd.merge(data['hol'], grouped, how='left', on=["year", "week"])
grouped.columns = ["year", "next_week", "next_week_hols"]
data['hol'] = pd.merge(data['hol'], grouped, how='left', on=["year", "next_week"])
grouped.columns = ["year", "prev_week", "prev_week_hols"]
data['hol'] = pd.merge(data['hol'], grouped, how='left', on=["year", "prev_week"])
data['hol']['day_delta'] = data['hol'].index
data['as']['air_store_num'] = le.fit_transform(data['as']['air_store_id'])
data['as']['air_genre_num'] = le.fit_transform(data['as']['air_genre_name'])
data['as']['air_area_num'] = le.fit_transform(data['as']['air_area_name'])
data['as']["prefecture"] = data['as']["air_area_name"].map(lambda x: str(x).split()[0])
data['as']["city"] = data['as']["air_area_name"].map(lambda x: str(x).split()[1])
data['as']["prefecture_num"] = le.fit_transform(data['as']['prefecture'])
data['as']["city_num"] = le.fit_transform(data['as']['city'])

train = pd.merge(data['train'], data['hol'], how='left', on=['visit_date'])
predict = pd.merge(data['predict'], data['hol'], how='left', on=['visit_date'])
train = pd.merge(train, data['as'], how='left', on=['air_store_id'])
predict = pd.merge(predict, data['as'], how='left', on=['air_store_id'])
train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

for df in ['ar', 'hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date'])
    predict = pd.merge(predict, data[df], how='left', on=['air_store_id', 'visit_date'])

train['total_reserve_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserve_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserve_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2
predict['total_reserve_sum'] = predict['rv1_x'] + predict['rv1_y']
predict['total_reserve_mean'] = (predict['rv2_x'] + predict['rv2_y']) / 2
predict['total_reserve_dt_diff_mean'] = (predict['rs2_x'] + predict['rs2_y']) / 2

rename_col = {
    "rv1_x": "reserve_sum_air",
    "rv2_x": "reserve_mean_air",
    "rs2_x": "reserve_datediff_mean_air",
    "rv1_y": "reserve_sum_hpg",
    "rv2_y": "reserve_mean_hpg",
    "rs2_y": "reserve_datediff_mean_hpg",
}
train = train.rename(columns=rename_col)
predict = predict.rename(columns = rename_col)

# print(train.describe())
# print(predict.describe())

hol_df = pd.read_csv("../input/date_info.csv")
hol_df = hol_df.rename(columns={"calendar_date": "next_date", "holiday_flg": "next_holiday_flg"})
hol_df["next_date_str"] = hol_df["next_date"].astype(str)


def get_more_dow_info(df, hol_df):
    df["next_datetime"] = df["visit_datetime"].dt.date + pd.DateOffset(days=1)
    df["next_date"] = df["next_datetime"].dt.date
    df["next_date_str"] = df["next_date"].astype(str)
    df = pd.merge(df, hol_df, how="left", on="next_date_str")
    df["next_dow"] = df["next_datetime"].dt.dayofweek
    df["next_dowh"] = np.where((df["next_holiday_flg"] == 1) & (df["next_dow"] < 5), 7, df["next_dow"])
    df["next_is_hol"] = np.where((df["next_holiday_flg"] == 1) | (df["next_dow"] >= 5), 1, 0)

    df["visit_datetime"] = pd.to_datetime(df["visit_date"])
    df["dowh"] = np.where((df["holiday_flg"] == 1) & (df["dow"] < 5), 7, df["dow"])
    df["dows"] = np.where(df["holiday_flg"] == 1, 6 - df["next_is_hol"], df["dow"])
    return df


train = get_more_dow_info(train, hol_df)
predict = get_more_dow_info(predict, hol_df)

train_col = ['air_store_num', 'visitors', 'visit_date', 'dow', 'day_delta',
             'year', 'month', 'prev_week', 'week', 'prev_month', 'prev_month_y', 'prev_month_m',
             'week_hols', 'next_week_hols', 'prev_week_hols', 'holiday_flg',
             'air_genre_num', 'air_area_num', "prefecture_num", "city_num",
             "reserve_sum_air", "reserve_mean_air", "reserve_datediff_mean_air",
             "reserve_sum_hpg", "reserve_mean_hpg", "reserve_datediff_mean_hpg",
             "total_reserve_sum", 'total_reserve_mean', 'total_reserve_dt_diff_mean',
             "next_dow", "next_is_hol",
             ]
predict_col = ['id', 'air_store_num', 'visitors', 'visit_date', 'dow', 'day_delta',
               'year', 'month', 'prev_week', 'week', 'prev_month', 'prev_month_y', 'prev_month_m',
               'week_hols', 'next_week_hols', 'prev_week_hols', 'holiday_flg',
               'air_genre_num', 'air_area_num', "prefecture_num", "city_num",
               "reserve_sum_air", "reserve_mean_air", "reserve_datediff_mean_air",
               "reserve_sum_hpg", "reserve_mean_hpg", "reserve_datediff_mean_hpg",
               "total_reserve_sum", 'total_reserve_mean', 'total_reserve_dt_diff_mean',
               "next_dow", 'next_is_hol',
               ]

# print(train.head())
# print(predict.head())

train = train[train_col]
predict = predict[predict_col]

train.to_csv('../output/cleaned_res_train.csv', float_format='%.6f', index=False)
predict.to_csv('../output/cleaned_res_predict.csv', float_format='%.6f', index=False)
