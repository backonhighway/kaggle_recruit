import numpy as np
import pandas as pd


train = pd.read_csv('../output/c_train.csv')
predict = pd.read_csv('../output/c_predict.csv')
joined = pd.concat([train, predict])


def get_weather(df):
    print('Adding weather...')
    air_nearest = pd.read_csv('../input/rrv-weather-data/air_store_info_with_nearest_active_station.csv')
    unique_air_store_ids = list(df["air_store_id"].unique())

    weather_dir = '../input/rrv-weather-data/weather/'
    weather_keep_columns = ["visit_date", 'precipitation', 'avg_temperature']

    weather_data_list = []
    for air_id in unique_air_store_ids:
        station_id = air_nearest[air_nearest["air_store_id"] == air_id]["station_id"].values[0]
        weather_data = pd.read_csv(weather_dir + station_id + '.csv').rename(columns={'calendar_date': 'visit_date'})
        weather_data = weather_data[weather_keep_columns]
        weather_data["air_store_id"] = air_id
        weather_data_list.append(weather_data)

    return weather_data_list


weather_list_df = get_weather(joined)
weather_df = pd.concat(weather_list_df)
print("now merging...")
joined = pd.merge(joined, weather_df, on=["air_store_id", "visit_date"], how="left")
train = joined[joined["visit_date"] <= "2017-04-22"]
predict = joined[joined["visit_date"] >= "2017-04-23"]

train.to_csv('../output/cw_train.csv', float_format='%.6f', index=False)
predict.to_csv('../output/cw_predict.csv', float_format='%.6f', index=False)