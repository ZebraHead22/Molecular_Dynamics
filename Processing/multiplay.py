import os
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt

IR_FILE = '/Users/max/Yandex.Disk.localized/Journals/FTIR/data/caf2/gly_spectre.dat'
MD_FILE = '/Users/max/Yandex.Disk.localized/NAMD/basic_ak_no_field/gly.dat'

IR_DATA = pd.read_csv(IR_FILE, delimiter=' ', index_col=None)
# IR_DATA.rename(columns={0: 'Frequency',
#                         1: 'Amplitude'}, inplace=True)
print(IR_DATA)
MD_DATA = pd.read_csv(MD_FILE, delimiter=' ', index_col=None)
MD_DATA.rename(columns={'0.0': 'Frequency',
                        '0.0.1': 'Amplitude'}, inplace=True)

IR_DATA["Frequency"] = [round(x) for x in IR_DATA["Frequency"].tolist()]
IR_DATA = IR_DATA.set_index("Frequency")
IR_DATA = IR_DATA.groupby(level='Frequency').mean()
IR_DATA = IR_DATA.reset_index() 
IR_DATA = IR_DATA.loc[(IR_DATA['Frequency'] >= 1000) & (IR_DATA['Frequency'] <= 5000)]

MD_DATA["Frequency"] = [round(x) for x in MD_DATA["Frequency"].tolist()]
MD_DATA = MD_DATA.set_index("Frequency")
MD_DATA = MD_DATA.groupby(level='Frequency').mean()
MD_DATA = MD_DATA.reset_index() 
MD_DATA = MD_DATA.loc[(MD_DATA['Frequency'] >= 1000) & (MD_DATA['Frequency'] <= 5000)]

if MD_DATA.shape[0] == IR_DATA.shape[0]:
    corr, pgly=stats.pearsonr(MD_DATA['Amplitude'], IR_DATA['Amplitude'])
    print(corr, pgly)
    df = pd.DataFrame()
    df['Frequency'] = MD_DATA["Frequency"]
    df['IR Amplitudes'] = IR_DATA['Amplitude']
    df['MD Amplitudes'] = MD_DATA['Amplitude']
    df['Correlation'] = (IR_DATA['Amplitude'] * MD_DATA['Amplitude'])*(10**4)
    print('Saving figure...')

    plt.gcf().clear()
    plt.plot(df['Frequency'].to_list(), df['Correlation'].tolist(), c = '#2D4354')
    plt.grid()
    plt.xlabel('Frequency ($cm^{-1}$)')
    plt.ylabel('Multiplication MD & IR Spectral Amplitude (a.u.) ×$10^{4}$')
    plt.xlim([-300, 6300])
    plt.savefig(os.getcwd()+'/gly_multiply.png')
else:
    print('Not equal\nMD_DATA: ' + str(MD_DATA.shape[0]) + ' & IR_DATA: ' + str(IR_DATA.shape[0]))
   