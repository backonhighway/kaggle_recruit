

def get_six_week_period_list():
    period_list = [
        ["2017-03-05", "2017-03-11"],
        ["2017-03-12", "2017-03-18"],
        ["2017-03-19", "2017-03-25"],
        ["2017-03-26", "2017-04-01"],
        ["2017-04-02", "2017-04-08"],
        ["2017-04-09", "2017-04-15"],
        ["2017-04-16", "2017-04-22"],
    ]
    return period_list


def get_six_week_df_list(df):
    df_list = []
    for period_from, period_to in get_six_week_period_list():
        temp = df[(df['visit_date'] >= period_from) & (df["visit_date"] <= period_to)]
        df_list.append(temp)

    return df_list

