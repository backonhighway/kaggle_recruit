import numpy as np
import pandas as pd

import custom_metrics
import custom_lgb
import pocket_split_train
import pocket_full_of_validator
import pocket_ez_validator
import pocket_periods


def get_submission(file_prefix):
    train_filename = file_prefix + 'cwrrsr_train.csv'
    test_filename = file_prefix + 'cwrrsr_predict.csv'
    train = pd.read_csv(train_filename)
    predict = pd.read_csv(test_filename)

    # make input
    train['visitors'] = np.log1p(train['visitors'])

    col = [
        'air_store_num', 'visitors', 'air_genre_num', "air_area_num",
        'year', 'month',
        "moving_mean_0", "moving_median_0", "moving_max_0", "moving_std_0",
        "moving_mean_1", "moving_median_1", "moving_max_1", "moving_std_1",
        "moving_mean_3", "moving_median_3", "moving_max_3", "moving_std_3",
        "moving_mean_13", "moving_median_13", "moving_max_13", "moving_std_13",
        "dow_moving_mean_0", "dow_moving_median_0", "dow_moving_max_0", "dow_moving_std_0",
        "dow_moving_mean_1", "dow_moving_median_1", "dow_moving_max_1", "dow_moving_std_1",
        "dow_moving_mean_3", "dow_moving_median_3", "dow_moving_max_3", "dow_moving_std_3",
        "dow_moving_mean_13", "dow_moving_median_13", "dow_moving_max_13", "dow_moving_std_13",
        "change_mean_0_1", "change_mean_0_3", "change_mean_0_13",  # small
        "change_mean_1_3", "change_mean_1_13", "change_mean_3_13",  # small
        "moving_skew_0", "moving_skew_1", "moving_skew_3", "moving_skew_13",
        "dow_moving_skew_1", "dow_moving_skew_3", "dow_moving_skew_13",
        'dow', 'dowh', 'dows', 'holiday_flg', 'week_hols', 'next_week_hols', 'prev_week_hols', "next_is_hol",
        'quarter_regress', 'year_regress', "ewm", 'quarter_regress_no_dow', 'year_regress_no_dow',
        "precipitation", "avg_temperature",
        "air_r_sum0_shifted", "hpg_r_sum0_shifted", "air_dow_r_sum0_shifted", "hpg_dow_r_sum0_shifted",
        "air_r_sum7", "hpg_r_sum7",
    ]

    # fit and predict
    period_list = [["2016-01-16", "2017-04-15", "2017-04-16", "2017-04-22"]
                   ]
    splits = pocket_split_train.split_set(train, period_list, col)
    models = pocket_split_train.split_train(splits)
    model = models[0]
    print("-" * 40)

    # train.drop("visit_date", axis=1, inplace=True)
    predict.drop("visit_date", axis=1, inplace=True)

    x_pred = predict[col].drop('visitors', axis=1)
    y_pred = model.predict(x_pred, num_iteration=model.best_iteration)
    y_pred = np.expm1(y_pred)
    y_pred[y_pred < 1] = 1

    fi = pd.DataFrame({"name": model.feature_name(), "importance": model.feature_importance()})
    fi = fi.sort_values(by="importance", ascending=False)
    print(fi.head())

    # submit
    submission = pd.DataFrame({
            "id": predict['id'],
            "visitors": y_pred,
        })
    submission["visit_date"] = submission['id'].map(lambda x: str(x).split('_')[2])
    print("Done one sub.")
    print("-"*30)

    return submission


# make submissions
w1_sub = get_submission("../output/w1_")
w2_sub = get_submission("../output/w2_")
w3_sub = get_submission("../output/w3_")
w4_sub = get_submission("../output/w4_")
w5_sub = get_submission("../output/w5_")

w1_sub = w1_sub[w1_sub["visit_date"] <= "2017-04-28"]
w2_sub = w2_sub[(w2_sub["visit_date"] >= "2017-04-29") & (w2_sub["visit_date"] <= "2017-05-06")]
w3_sub = w3_sub[(w3_sub["visit_date"] >= "2017-05-07") & (w3_sub["visit_date"] <= "2017-05-13")]
w4_sub = w4_sub[(w4_sub["visit_date"] >= "2017-05-14") & (w4_sub["visit_date"] <= "2017-05-20")]
w5_sub = w5_sub[(w5_sub["visit_date"] >= "2017-05-21")]
print("-"*30)
print(w1_sub.describe())
print("-"*30)
print(w2_sub.describe())
print("-"*30)
print(w3_sub.describe())
print("-"*30)
print(w4_sub.describe())
print("-"*30)
print(w5_sub.describe())
print("-"*30)

sub_list = [w1_sub, w2_sub, w3_sub, w4_sub, w5_sub]
final_sub = pd.concat(sub_list)
print(final_sub.describe())

col = ["id", "visitors"]
final_sub = final_sub[col]
final_sub.to_csv('../output/submission_final.csv', float_format='%.6f', index=False)
