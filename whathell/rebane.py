import numpy as np
import pandas as pd

import custom_lgb

# load data
train = pd.read_csv('../output/cleaned_train.csv')
predict = pd.read_csv('../output/cleaned_predict.csv')

print(train.head())
print(predict.head())

# make input
train['visitors'] = np.log1p(train['visitors'])
train_input = train[ (train['visit_date_str'] >= '2016-01-01') & (train['visit_date_str'] < '2016-12-01') ].reset_index(drop=True)
test_input = train[ (train['visit_date_str'] >= '2017-03-01') & (train['visit_date_str'] < '2017-04-01') ].reset_index(drop=True)

col = ['air_store_num', 'visitors', 'dow', 'holiday_flg', 'air_genre_num', 'air_area_num']
train_input = train_input[col]
test_input = test_input[col]
x_pred = predict[col].drop('visitors', axis=1)

# print(train_input.head())
# print(test_input.head())
# print(x_pred.head())
# print('-' * 30)

# fit and predict
model = custom_lgb.do_predict(train_input, test_input, x_pred)
y_pred = model.predict(x_pred, num_iteration=model.best_iteration)
# y_pred[y_pred < 0] = 0
y_pred = np.expm1(y_pred)

# validate
validate_data = train[ train['visit_date_str'] >= '2017-04-01' ].reset_index(drop=True)



# submit
submission = pd.DataFrame({
        "id": predict['id'],
        "visitors": y_pred
    })
print(submission.describe())
submission.to_csv('../output/submission.csv',float_format='%.6f', index=False)


