import pandas as pd


normal_sub = pd.read_csv('../output/submission_final_daily.csv')
out_sub = pd.read_csv('../output/submission_final_daily_outlier.csv')

togethered = pd.merge(normal_sub, out_sub, how="left", on="id", suffixes=["_normal", "_out"])

togethered["visitors"] = togethered["visitors_normal"] * 0.5 + togethered["visitors_out"] * 0.5
togethered["visit_date"] = togethered['id'].map(lambda x: str(x).split('_')[2])
print(togethered.describe(include="all"))
print(togethered.groupby("visit_date")["visitors"].describe())

togethered = togethered[["id", "visitors"]]
togethered.to_csv('../output/submission_final_daily_ensembled.csv', float_format='%.6f', index=False)