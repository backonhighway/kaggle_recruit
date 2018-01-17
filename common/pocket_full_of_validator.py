import custom_metrics
import pandas as pd


# validate
def validate(test_df, model, col, period_from, period_to):
    validate_data = test_df[(test_df['visit_date'] >= period_from) & (test_df["visit_date"] <= period_to)]
    x_valid = validate_data[col].drop('visitors', axis=1)
    y_valid = model.predict(x_valid)
    validation_score = custom_metrics.rmse(validate_data["visitors"], y_valid)
    print(validation_score)


def dow_analyze(test_df, model, col, period_from, period_to):
    validate_data = test_df[(test_df['visit_date'] >= period_from) & (test_df["visit_date"] <= period_to)]
    validate_data = validate_data[col]

    score_list = []
    for dow_num in range(7):
        print("-" * 40)
        print("dow= ", dow_num)
        dowed_data = validate_data[validate_data["dow"] == dow_num]
        print(dowed_data["visitors"].describe())

        x_valid = dowed_data.drop('visitors', axis=1)
        y_valid = model.predict(x_valid)
        print(pd.DataFrame(y_valid).describe())
        validation_score = custom_metrics.rmse(dowed_data["visitors"], y_valid)
        score_list.append(validation_score)
    print(score_list)

def day_by_day_analyze(test_df, model, col, period_from, period_to):

    days = pd.date_range(start=period_from, end=period_to, freq='D')
    for day in days:
        print("day=", day)
