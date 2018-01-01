import numpy as np
import pandas as pd

import custom_metrics
import custom_lgb
import pocket_split_train

# load data
train = pd.read_csv('../output/cleaned_train.csv')
predict = pd.read_csv('../output/cleaned_predict.csv')

# make input
train['visitors'] = np.log1p(train['visitors'])
period_split = [
    ['2016-01-10', '2016-04-23', '2016-04-24', '2016-05-31'],
    ['2016-01-10', '2016-09-03', '2016-09-04', '2016-10-08'],
    ['2016-01-10', '2016-11-26', '2017-01-16', '2017-03-05'],
]
models, score = pocket_split_train.split_train(train, period_split)

# print(train_input.head())
# print(test_input.head())
# print(x_pred.head())
# print('-' * 30)

exit(0)
