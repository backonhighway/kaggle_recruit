import pandas as pd


# load data
my_model = pd.read_csv('../output/submission.csv')
kernel_model = pd.read_csv('../output/h2o_submission.csv')

print(my_model.describe())
print(kernel_model.describe())

my_model['visit_date_str'] = my_model['id'].map(lambda x: str(x).split('_')[2])
my_model['air_store_id'] = my_model['id'].map(lambda x: '_'.join(x.split('_')[:2]))
kernel_model['visit_date_str'] = kernel_model['id'].map(lambda x: str(x).split('_')[2])
kernel_model['air_store_id'] = kernel_model['id'].map(lambda x: '_'.join(x.split('_')[:2]))

my_model["visit_dow"] = pd.to_datetime(my_model["visit_date_str"]).dt.dayofweek
grouped = my_model.groupby("visit_dow")
print(grouped.describe())

kernel_model["visit_dow"] = pd.to_datetime(kernel_model["visit_date_str"]).dt.dayofweek
grouped = kernel_model.groupby("visit_dow")
print(grouped.describe())

averaged_model = pd.merge(my_model, kernel_model, on="id", how="inner")
averaged_model["visitors"] = (averaged_model["visitors_x"] + averaged_model["visitors_y"]) / 2

print("-"*40)
print(averaged_model.describe())
averaged_model[['id', 'visitors']].to_csv('../output/averaged.csv', index=False)
