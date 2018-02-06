import numpy as np
import pandas as pd

import custom_metrics
import custom_lgb
import pocket_split_train
import pocket_full_of_validator
import pocket_ez_validator
import pocket_periods


def get_submission(file_prefix, reserve_col):
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
        "dow_moving_mean_0", "dow_moving_median_0", "dow_moving_max_0",
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
        #"air_r_sum0_shifted", "hpg_r_sum0_shifted", "air_dow_r_sum0_shifted", "hpg_dow_r_sum0_shifted",
        #"air_r_sum7", "hpg_r_sum7",
    ]
    col.extend(reserve_col)

    # fit and predict
    period_list = [["2016-01-16", "2017-04-15", "2017-04-16", "2017-04-22"]
                   ]
    splits = pocket_split_train.split_set(train, period_list, col)
    models = pocket_split_train.split_train(splits)
    model = models[0]

    good_stores = train[train["first_appear"] < "2016-04-01"]
    good_store_list = good_stores["air_store_id"].unique()
    splits = pocket_split_train.split_set(good_stores, period_list, col)
    models = pocket_split_train.split_train(splits)
    good_store_model = models[0]
    print("-" * 40)

    # train.drop("visit_date", axis=1, inplace=True)
    predict.drop("visit_date", axis=1, inplace=True)

    x_pred = predict[col].drop('visitors', axis=1)
    y_pred = model.predict(x_pred, num_iteration=model.best_iteration)
    y_pred = np.expm1(y_pred)
    y_pred[y_pred < 1] = 1

    y_pred_good = good_store_model.predict(x_pred, num_iteration=good_store_model.best_iteration)
    y_pred_good = np.expm1(y_pred_good)
    y_pred_good[y_pred_good < 1] = 1

    #fi = pd.DataFrame({"name": model.feature_name(), "importance": model.feature_importance()})
    #fi = fi.sort_values(by="importance", ascending=False)
    #print(fi.head())

    # submit
    submission = pd.DataFrame({
            "id": predict['id'],
            "v_good": y_pred_good,
            "v_all": y_pred,
        })
    submission["air_store_id"] = submission['id'].map(lambda x: '_'.join(x.split('_')[:2]))
    submission["is_good_store"] = submission["air_store_id"].isin(good_store_list)
    submission["visit_date"] = submission['id'].map(lambda x: str(x).split('_')[2])
    submission["visitors"] = np.where(submission["is_good_store"], submission["v_good"], submission["v_all"])
    print("Done one sub.")
    print("-"*30)

    return submission


base_res_col = [
    "air_r_sum0_shifted", "hpg_r_sum0_shifted", "air_dow_r_sum0_shifted", "hpg_dow_r_sum0_shifted",
]
d7_col = base_res_col + ["air_r_sum7", "hpg_r_sum7"]
d8_col = base_res_col + ["air_r_sum8", "hpg_r_sum8"]
d9_col = base_res_col + ["air_r_sum9", "hpg_r_sum9"]
d10_col = base_res_col + ["air_r_sum10", "hpg_r_sum10"]
d11_col = base_res_col + ["air_r_sum11", "hpg_r_sum11"]
d12_col = base_res_col + ["air_r_sum12", "hpg_r_sum12"]
d13_col = base_res_col + ["air_r_sum13", "hpg_r_sum13"]
d14_col = base_res_col + ["air_r_sum14", "hpg_r_sum14"]
weekly_col = base_res_col + ["air_r_sum7", "hpg_r_sum7"]

# make submissions
w1_sub = get_submission("../output/outlier_w1_", weekly_col)
w2_sub = get_submission("../output/outlier_w2_", weekly_col)
w3_sub = get_submission("../output/outlier_w3_", weekly_col)
w4_sub = get_submission("../output/outlier_w4_", weekly_col)
w5_sub = get_submission("../output/outlier_w5_", weekly_col)
print("Start making daily...")
print("-"*30)
d7_sub = get_submission("../output/outlier_w2_", d7_col)
d8_sub = get_submission("../output/outlier_w2_", d8_col)
d9_sub = get_submission("../output/outlier_w2_", d9_col)
d10_sub = get_submission("../output/outlier_w2_", d10_col)
d11_sub = get_submission("../output/outlier_w2_", d11_col)
d12_sub = get_submission("../output/outlier_w2_", d12_col)
d13_sub = get_submission("../output/outlier_w2_", d13_col)
d14_sub = get_submission("../output/outlier_w2_", d14_col)


def take_df(target_df_to_cut, day):
    return target_df_to_cut[target_df_to_cut["visit_date"] == day]


d7_sub = take_df(d7_sub, "2017-04-29")
d8_sub = take_df(d8_sub, "2017-04-30")
d9_sub = take_df(d9_sub, "2017-05-01")
d10_sub = take_df(d10_sub, "2017-05-02")
d11_sub = take_df(d11_sub, "2017-05-03")
d12_sub = take_df(d12_sub, "2017-05-04")
d13_sub = take_df(d13_sub, "2017-05-05")
d14_sub = take_df(d14_sub, "2017-05-06")

sub_list = [d7_sub, d8_sub, d9_sub, d10_sub, d11_sub, d12_sub, d13_sub, d14_sub]
w2_daily_sub = pd.concat(sub_list)
print("-"*30)
print(w2_daily_sub.describe())

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

w2_final = pd.merge(w2_sub, w2_daily_sub, how="left", on="id", suffixes=["_weekly", "_daily"])
w2_final["visitors"] = w2_final["visitors_daily"] * 0.7 + w2_final["visitors_weekly"] * 0.3
print("w2_final:")
print(w2_final.describe())

sub_list = [w1_sub, w2_final, w3_sub, w4_sub, w5_sub]
final_sub = pd.concat(sub_list)
print("final submission is:")
print(final_sub.groupby("visit_date")["visitors"].describe())

submission_col = ["id", "visitors"]
final_sub = final_sub[submission_col]
final_sub.to_csv('../output/submission_final_daily_outlier.csv', float_format='%.6f', index=False)







