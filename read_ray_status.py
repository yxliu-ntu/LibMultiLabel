from ray.tune import Analysis
import os, sys

analysis = Analysis(sys.argv[1])
df = analysis.dataframe().sort_values('val_P@1')
tmp = df.iloc[-1]
for i in tmp.keys():
    print(i, tmp[i])
