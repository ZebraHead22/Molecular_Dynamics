import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Тут строим старые спектры, от 060224. Тут есть повторения точек. Странные деления
def ir_spectres():
    gly_data = pd.DataFrame()
    val_data = pd.DataFrame()
    trp_data = pd.DataFrame()
    ala_data = pd.DataFrame()

    folder = os.getcwd()
    files = os.listdir(folder)
    for file in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+file)
        if file_extension == ".dpt":
            if os.path.basename(filename) == 'Tryptophan':
                df = pd.read_csv(os.getcwd()+'/'+file,
                                 delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                                   1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                trp_data.insert(0, 'Frequency_'+str(os.path.basename(filename)),
                                df['Frequency_'+str(os.path.basename(filename))])
                trp_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)),
                                df['Amplitude_'+str(os.path.basename(filename))])
            elif os.path.basename(filename) == 'Alanine':
                df = pd.read_csv(os.getcwd()+'/'+file,
                                 delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                                   1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                ala_data.insert(0, 'Frequency_'+str(os.path.basename(filename)),
                                df['Frequency_'+str(os.path.basename(filename))])
                ala_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)),
                                df['Amplitude_'+str(os.path.basename(filename))])
            elif os.path.basename(filename) == 'Glycine':
                df = pd.read_csv(os.getcwd()+'/'+file,
                                 delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                                   1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                gly_data.insert(0, 'Frequency_'+str(os.path.basename(filename)),
                                df['Frequency_'+str(os.path.basename(filename))])
                gly_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)),
                                df['Amplitude_'+str(os.path.basename(filename))])
            else:
                df = pd.read_csv(os.getcwd()+'/'+file,
                                 delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                                   1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                val_data.insert(0, 'Frequency_'+str(os.path.basename(filename)),
                                df['Frequency_'+str(os.path.basename(filename))])
                val_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)),
                                df['Amplitude_'+str(os.path.basename(filename))])

    trp_data = trp_data.loc[(trp_data['Frequency_Tryptophan'] >= 500) & (
        trp_data['Frequency_Tryptophan'] <= 5000)]
    ala_data = ala_data.loc[(ala_data['Frequency_Alanine'] >= 500) & (
        ala_data['Frequency_Alanine'] <= 5000)]
    val_data = val_data.loc[(val_data['Frequency_Valine'] >= 500) & (
        val_data['Frequency_Valine'] <= 5000)]
    gly_data = gly_data.loc[(gly_data['Frequency_Glycine'] >= 500) & (
        gly_data['Frequency_Glycine'] <= 5000)]

    plt.plot(val_data["Frequency_Valine"], val_data["Amplitude_Valine"])
    plt.plot(gly_data["Frequency_Glycine"], gly_data["Amplitude_Glycine"])
    plt.plot(ala_data["Frequency_Alanine"], ala_data["Amplitude_Alanine"])
    plt.plot(trp_data["Frequency_Tryptophan"],
             trp_data["Amplitude_Tryptophan"])
    plt.legend(["Valine", "Glycine", "Alanine", "Tryptophan"])
    plt.grid()
    plt.xlim(-300, 6300)
    plt.ylim(0, 1)
    plt.xlabel("Frequency ($cm^{-1}$)")
    plt.ylabel("Energy (a.u.)")
    plt.savefig(folder + "/result.png")

def caf2_plot():
    folder = os.getcwd()
    files = os.listdir(folder)
    dfs = pd.read_csv(os.getcwd()+'/surface.dpt', delimiter=',', index_col=None, header=None)
    dfs.rename(columns={0: 'Frequency', 1: 'Amplitude'}, inplace=True)
    # dfs = dfs.loc[(dfs['Frequency'] <= 5000)]
    plt.plot(dfs["Frequency"], dfs["Amplitude"], c='black')
    plt.grid()
    plt.title('CaF2 surface')
    plt.savefig(folder + "/caf2_surface.png")
    plt.gcf().clear()

    legend = list()
    for file in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+file)
        if file_extension == ".dpt":
            if 'surface' not in os.path.basename(filename):
                legend.append(os.path.basename(filename).upper())
                df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency', 1: 'Amplitude'}, inplace=True)
                print(df.head())
                
                df.insert(2, 'Amplitude_wo', 1 - (np.array(df['Amplitude'].to_list())/np.array(dfs['Amplitude'].to_list())))
                df = df.loc[(df['Frequency'] >= 1000) & (df['Frequency'] <= 5000)]
                
                max_amp = df.loc[df['Amplitude_wo'].idxmax(), 'Amplitude_wo']
                min_amp = df.loc[df['Amplitude_wo'].idxmin(), 'Amplitude_wo']
                
                df['Amplitude_wo'] = df['Amplitude_wo'] - min_amp
                
                #Save .dat files
                # output_data = np.column_stack((df['Frequency'].to_list(), df['Amplitude_wo'].to_list()))
                # output_file_path = filename + '_spectre.dat'
                # np.savetxt(output_file_path, output_data, fmt='%.6e', delimiter=' ', header='Frequency Amplitude', comments='')
                
                # Graphs
                # plt.gcf().clear()
                plt.plot(df["Frequency"], df["Amplitude_wo"])
    plt.legend(legend)  
    plt.grid()
    plt.xlabel("Frequency ($cm^{-1}$)")
    plt.ylabel("Energy (a.u.)")
                # plt.title(str(os.path.basename(filename)).upper())      
    plt.savefig(filename + "1.png")

def kbr_plot():
    from pathlib import Path
    surface_data = pd.DataFrame()
    folder = os.getcwd()
    files = os.listdir(folder)
    path = Path(folder)
    count = 1

    for f in path.glob("surface*.dpt"):
        df = pd.read_csv(f, delimiter=',', index_col=None, header=None)
        df.rename(columns={0: 'Frequency', 1: 'Amplitude'}, inplace=True)
        surface_data['Frequency'] = df['Frequency']
        surface_data['Amplitude_' + str(count)] = df['Amplitude']
        count += 1

    surface_data['Mean Amplitude'] = surface_data.iloc[:, 1:4].mean(axis=1)
    # surface_data = surface_data.loc[(surface_data['Frequency'] <= 5000)]

    #Save .dat files
    # output_data = np.column_stack((df['Frequency'].to_list(), df['Amplitude'].to_list()))
    # output_file_path = os.getcwd() + '/KBr.dpt'
    # np.savetxt(output_file_path, output_data, fmt='%.6e', delimiter=' ', header='Frequency Amplitude', comments='')
    vacuum_file = folder + '/vacuum.dpt'
    vacuum_data = pd.read_csv(vacuum_file, delimiter=',', index_col=None, header=None)
    vacuum_data.rename(columns={0: 'Frequency', 1: 'Amplitude'}, inplace=True)

    plt.gcf().clear()
    plt.plot(surface_data["Frequency"], surface_data["Mean Amplitude"], c='black')
    plt.grid()
    plt.title('KBr Surface')
    plt.xlim([0, 6000])
    plt.savefig(folder + "/KBr_surcafe.png")

    plt.gcf().clear()
    plt.plot(vacuum_data["Frequency"], vacuum_data["Amplitude"], c='black')
    plt.grid()
    plt.title('Vacuum')
    plt.xlim([0, 6000])
    plt.savefig(folder + "/vacuum.png")
    plt.gcf().clear()

    legend = list()

    for file in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+file)
        if file_extension == ".dpt":
            if 'surface' not in os.path.basename(filename) and 'vacuum' not in os.path.basename(filename):
                legend.append(os.path.basename(filename).upper())
                df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency', 1: 'Amplitude'}, inplace=True)

                df.insert(2, 'Amplitude_wo', 1 - (np.array(df['Amplitude'].to_list())/np.array(surface_data['Mean Amplitude'].to_list())))
                df = df.loc[(df['Frequency'] >= 1000) & (df['Frequency'] <= 5000)]
                
                max_amp = df.loc[df['Amplitude_wo'].idxmax(), 'Amplitude_wo']
                min_amp = df.loc[df['Amplitude_wo'].idxmin(), 'Amplitude_wo']

                df['Amplitude_wo'] = df['Amplitude_wo'] - min_amp

                #Save .dat files
                # output_data = np.column_stack((df['Frequency'].to_list(), df['Amplitude_wo'].to_list()))
                # output_file_path = filename + '_spectre.dat'
                # np.savetxt(output_file_path, output_data, fmt='%.6e', delimiter=' ', header='Frequency Amplitude', comments='')
                
                # Graphs
                # plt.gcf().clear()
                plt.plot(df["Frequency"], df["Amplitude_wo"])
    plt.legend(legend)
    plt.grid()
    plt.xlabel("Frequency ($cm^{-1}$)")
    plt.ylabel("Energy (a.u.)")
                # plt.title(str(os.path.basename(filename)).upper())
    plt.savefig(filename + "1.png")
    print(legend)

def si_plot():
    folder = os.getcwd()
    files = os.listdir(folder)
    dfs = pd.read_csv(os.getcwd()+'/surface.dpt', delimiter=',', index_col=None, header=None)
    dfs.rename(columns={0: 'Frequency', 1: 'Amplitude'}, inplace=True)
    dfs = dfs.loc[(dfs['Frequency'] <= 5000)]
    plt.plot(dfs["Frequency"], dfs["Amplitude"], c='black')
    plt.grid()
    plt.title('Si surface')
    plt.savefig(folder + "/Si_surface.png")
    plt.gcf().clear()

    legend = list()
    for file in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+file)
        if file_extension == ".dpt":
            if 'surface' not in os.path.basename(filename):
                legend.append(os.path.basename(filename).upper())
                df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency', 1: 'Amplitude'}, inplace=True)
                print(df.head())
                
                df.insert(2, 'Amplitude_wo', 1 - (np.array(df['Amplitude'].to_list())))
                df = df.loc[(df['Frequency'] >= 1000) & (df['Frequency'] <= 5000)]
                
                max_amp = df.loc[df['Amplitude_wo'].idxmax(), 'Amplitude_wo']
                min_amp = df.loc[df['Amplitude_wo'].idxmin(), 'Amplitude_wo']
                
                df['Amplitude_wo'] = df['Amplitude_wo'] - min_amp
                
                #Save .dat files
                # output_data = np.column_stack((df['Frequency'].to_list(), df['Amplitude_wo'].to_list()))
                # output_file_path = filename + '_spectre.dat'
                # np.savetxt(output_file_path, output_data, fmt='%.6e', delimiter=' ', header='Frequency Amplitude', comments='')
                
                # Graphs
                # plt.gcf().clear()
                plt.plot(df["Frequency"], df["Amplitude_wo"])
    # plt.legend(legend)  
    plt.grid()
    plt.xlabel("Frequency ($cm^{-1}$)")
    plt.ylabel("Energy (a.u.)")
    plt.legend(legend)
    # plt.title(str(os.path.basename(filename)).upper())
    plt.savefig(filename + "1.png")


def surface():
    folder = os.getcwd()
    files = os.listdir(folder)
    legend = list()
    for file in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+file)
        if file_extension == ".dpt":
            legend.append(os.path.basename(filename))
            df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
            df.rename(columns={0: 'Frequency', 1: 'Amplitude'}, inplace=True)
            df = df.loc[(df['Frequency'] <= 5000)]
            plt.plot(df["Frequency"], df["Amplitude"])
    plt.grid()
    plt.legend(legend)
    plt.xlabel("Frequency ($cm^{-1}$)")
    plt.ylabel("Energy (a.u.)")
    plt.savefig(folder + "/surfaces.png")

kbr_plot()
# caf2_plot()
# si_plot()
# surface()