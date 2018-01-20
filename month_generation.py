import pandas as pd
import pandas.tseries.offsets as offsets

week_ends = pd.date_range(start='01/02/2017', end='05/01/2017', freq='W')
print(week_ends)
print(week_ends[16] + offsets.Week(1))

exit(0)


month_ends = pd.date_range(start='01/01/2016', end='05/01/2017', freq='M')
print(month_ends[15].replace(day=22))
print(month_ends[15])
if month_ends[15].month == 4 and month_ends[15].year == 2017:
    month_ends[15].replace(day=22)
    print(month_ends[15])
print(month_ends)
exit(0)
for month_end in month_ends:
    quarter_start = month_end - offsets.MonthBegin(3)
    next_month_start = month_end + offsets.MonthBegin(1)
    next_month_end = month_end + offsets.MonthEnd(1)
    year_start = month_end - offsets.MonthBegin(12)

    print("-"*30)
    print(quarter_start)
    print(next_month_start)
    print(next_month_end)
    print(year_start)
