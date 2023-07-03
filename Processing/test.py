import pandas as pd

df_1 = pd.read_csv("E:/namd\spectres_high_frequency_fields\gly/gly_0.dat", delimiter=' ', index_col=None)
df_1.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude_1'}, inplace=True)

df_2 = pd.read_csv("E:/namd\spectres_high_frequency_fields\gly/gly_4167.dat", delimiter=' ', index_col=None)
df_2.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude_2'}, inplace=True)

data = pd.merge(df_1, df_2, how='inner', left_index=True, right_index=True)
print(data.head(20))
# 
