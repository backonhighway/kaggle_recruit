import numpy as np
import pandas as pd
import glob, re

# originally from this kernel
# https://www.kaggle.com/guoqiangliang/median-by-dow-lb-0-517
test_df = pd.read_csv('./input/sample_submission.csv')
test_df['store_id'], test_df['visit_date'] = test_df['id'].str[:20], test_df['id'].str[21:]
test_df.drop(['visitors'], axis=1, inplace=True)
test_df['visit_date'] = pd.to_datetime(test_df['visit_date'])

air_data = pd.read_csv('../input/air_visit_data.csv', parse_dates=['visit_date'])
air_data['dow'] = air_data['visit_date'].dt.dayofweek
train = air_data[air_data['visit_date'] > '2017-01-28'].reset_index()
train['dow'] = train['visit_date'].dt.dayofweek
test_df['dow'] = test_df['visit_date'].dt.dayofweek
aggregation = {'visitors': {'total_visitors': 'median'}}

# Group by id and day of week - Median of the visitors is taken
agg_data = train.groupby(['air_store_id', 'dow']).agg(aggregation).reset_index()
agg_data.columns = ['air_store_id', 'dow', 'visitors']
agg_data['visitors'] = agg_data['visitors']

# Create the first intermediate submission file:
merged = pd.merge(test_df, agg_data, how='left', left_on=[
    'store_id', 'dow'], right_on=['air_store_id', 'dow'])
final = merged[['id', 'visitors']]
final.fillna(0, inplace=True)

sample_submission = pd.read_csv('./input/sample_submission.csv')
sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
sample_submission = sample_submission[['id', 'visitors']]
final['visitors'][final['visitors'] == 0] = sample_submission['visitors'][final['visitors'] == 0]
sub_file = final.copy()

## Arithmetric Mean
sub_file['visitors'] = np.mean([final['visitors'], sample_submission['visitors']], axis=0)
sub_file.to_csv('sub_math_mean.csv', index=False)

## Geometric Mean
sub_file['visitors'] = (final['visitors'] * sample_submission['visitors']) ** (1 / 2)
sub_file.to_csv('sub_geo_mean.csv', index=False)

## Harmonic Mean
sub_file['visitors'] = 2 / (1 / final['visitors'] + 1 / sample_submission['visitors'])
sub_file.to_csv('sub_hrm_mean.csv', index=False)