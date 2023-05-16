import pandas as pd
import pandasx_time as pdxt
# B         business day frequency
# C         custom business day frequency
# D         calendar day frequency
# W         weekly frequency
# M         month end frequency
# SM        semi-month end frequency (15th and end of month)
# BM        business month end frequency
# CBM       custom business month end frequency
# MS        month start frequency
# SMS       semi-month start frequency (1st and 15th)
# BMS       business month start frequency
# CBMS      custom business month start frequency
# Q         quarter end frequency
# BQ        business quarter end frequency
# QS        quarter start frequency
# BQS       business quarter start frequency
# A, Y      year end frequency
# BA, BY    business year end frequency
# AS, YS    year start frequency
# BAS, BYS  business year start frequency
# BH        business hour frequency
# H         hourly frequency

ts = pd.Timestamp('2020-03-14T15:32:52.192548')
print(ts.timestamp())
ts = pd.Timestamp('2020-03-14 15:32:52.192548')
print(ts.timestamp())


print('---')
for freq in pdxt.FREQUENCIES:
    try:
        print(freq, ':', pd.period_range("2024-01-01", periods=20, freq=freq))
    except:
        print(freq, ': Unsupported')
# end

print('---')
for freq in pdxt.FREQUENCIES:
    try:
        print(freq, ':', pd.period_range("2024-01-31", periods=20, freq=freq))
    except:
        print(freq, ': Unsupported')
# end
