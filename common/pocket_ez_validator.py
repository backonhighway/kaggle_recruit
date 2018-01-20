import custom_metrics
import pocket_periods


def validate(df, actual_col_name, validate_col_name, period_list=None, verbose=False):
    if period_list is None:
        period_list = pocket_periods.get_six_week_period_list()

    score_list = []
    for period_from, period_to in period_list:
        validate_data = df[(df['visit_date'] >= period_from) & (df["visit_date"] <= period_to)]
        validation_score = custom_metrics.rmse(validate_data[actual_col_name], validate_data[validate_col_name])
        score_list.append(validation_score)
        if verbose:
            print("score from ", period_from, " to ", period_to, " is:")
            print(validation_score)

    print("average score for", validate_col_name, " is:")
    average_score = sum(score_list) / float(len(score_list))
    print(average_score)
    print("-" * 40)