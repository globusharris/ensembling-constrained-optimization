from datasets import load_dataset
import pandas as pd
from functools import reduce


id = "gauss314/options-IV-SP500"
data_iv = load_dataset(id)
df_iv = pd.DataFrame(data_iv['train'][:])


df_iv = df_iv.drop(["Unnamed: 0"], axis=1)
names = df_iv["symbol"].unique()
# names is the subset of option symbols for the options that we keep data for
# can modify this list so that it is only a subset of options, if desired

splits = []
for i in names:
 splits.append(df_iv.loc[df_iv["symbol"] == i])
for i in range(len(splits)):
 ticker = splits[i].iloc[0]["symbol"]
 splits[i] = splits[i].add_suffix("_" + ticker)
 splits[i] = splits[i].drop(["symbol" + "_" + ticker], axis=1)
 splits[i].rename(columns={"date" + "_" + ticker: "date"}, inplace=True)


 # just keeping options with no missing data
large = []
for i in splits:
 if len(i) >= 938:
  large.append(i)

final = reduce(lambda x, y: pd.merge(x, y, on = "date"), large)
final.to_csv("options500_v2")