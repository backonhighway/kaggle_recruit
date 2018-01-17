import pandas as pd


days = pd.date_range(start="2017-04-15", end="2017-04-22", freq='D')
for day in days:
    print("day=", day)
