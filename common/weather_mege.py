import glob
import pandas as pd


#dfs = {re.search('/([^/\.]*)\.csv', fn).group(1):
#           pd.read_csv(fn) for fn in glob.glob('../input/*.csv')}
path = "../input/rrv-weather-data/weathers"
all_files = glob.glob(path + "/*.csv")
list_ = []
for file_ in all_files:
    df = pd.read_csv(file_)
    df["station_id"] = file_
    # df["station_id"]
    list_.append(df)
df = pd.concat(list_)
print(df.info())

df.to_csv('../input/merged_weather.csv', index=False)
# df.to_csv('../input/merged_weather.csv', float_format='%.6f', index=False)

train = pd.read_csv('../output/cleaned_res_train.csv')
