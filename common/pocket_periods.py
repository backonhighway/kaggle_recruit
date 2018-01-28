
def get_prev_year_period_list():
    period_list = [
        ["2016-04-23", "2016-04-28"],
        ["2016-04-29", "2016-05-06"],
        ["2016-05-07", "2016-05-13"],
        ["2016-05-14", "2016-05-20"],
        ["2016-05-21", "2016-05-27"],
        ["2016-05-28", "2016-05-31"],
    ]
    return period_list


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


def get_df_list(df, period_list):
    df_list = []
    for period_from, period_to in period_list:
        temp = df[(df['visit_date'] >= period_from) & (df["visit_date"] <= period_to)]
        df_list.append(temp)

    return df_list
