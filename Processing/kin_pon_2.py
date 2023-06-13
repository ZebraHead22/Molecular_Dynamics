import os
import re
import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy import fftpack
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import LinearLocator, FormatStrFormatter

amino_acids = list()
energy_type = ['KINETIC', 'POTENTIAL']
field_amplitudes = [0.0435,	0.087,	0.1305,	0.174,	0.2175,
                    0.261,	0.3045,	0.348,	0.3915,	0.435,	0.4785,	0.522]

data = pd.DataFrame()
directory = os.getcwd()
for address, dirs, names in os.walk(directory):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            amino_acid = re.search(r'^\w{,2}[^\_]', filename)
            amino_acid = amino_acid.group(0)
            amino_acids.append(amino_acid)
amino_acids = list(set(amino_acids))

for i in amino_acids:
    directory = os.getcwd()
    directory = directory+"/"+i
    for j in energy_type:
        data = pd.DataFrame()
        for address, dirs, names in os.walk(directory):
            for name in names:
                filename, file_extension = os.path.splitext(name)
                if file_extension == ".dat":
                    one_data = pd.read_csv(os.path.join(
                        address, name), delimiter=' ', index_col=None, header=[0])
                    data["TIME"] = (one_data["TS"]) * 0.001
                    data[(str(filename)+"_"+j)] = (one_data[j]) * 0.0434/5400
        # Тут делаем график 1 и 2
        plt.gcf().clear()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
       
        def cc(arg):
            return mcolors.to_rgba(arg, alpha=0.6)

        xs = np.array(data["TIME"].tolist())
        verts = []
        ys = field_amplitudes
        a = 1
        for y in ys:
            zs = data.iloc[:,a].tolist()
            verts.append(list(zip(xs, zs)))
            a+=1
        print(len(verts))
      
        poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
                                                cc('y')])
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(i+'_'+j)
        plt.show()
        # plt.savefig(directory+'/'+filename+'_'+j+'.png')


#         filename, file_extension = os.path.splitext(name)
#         if file_extension == ".dat":
#             one_data = pd.read_csv(os.path.join(
#                 address, name), delimiter=' ', index_col=None, header=[0])

#             # one_data.rename(columns={0: "TS", 1: "Amplitude"}, inplace=True)
#             one_data.insert(1, "Time", (one_data["TS"]) * 0.001)
#             one_data.insert(3, "Kinetic", (one_data["KINETIC"] * 0.0434/5400))
#             one_data.insert(3, "Potential", (one_data["POTENTIAL"] * 0.0434/5400))
#             data["Time"] = one_data["Time"]
#             # data[(str(filename)+"_Kinetic")] = one_data["Kinetic"]
#             data[(str(filename)+"_Potential")] = one_data["Potential"]
#             print(filename)
# data.to_excel(directory+"/trp_potential.xlsx")

"""Построение семейтв энергий в двух окнах"""
# fig, ax = plt.subplots(nrows=1, ncols=2)

# def cm_to_inch(value):  # Define picture size, calc cm in inch
#     return value/2.54

# fig.set_figheight(cm_to_inch(10))
# fig.set_figwidth(cm_to_inch(30))

# ax[0].scatter(data["2_kin_Time"].tolist(),
#               data["2_kin_Energy"].tolist(), c='#FFDD19', s=2)
# ax[0].scatter(data["4_kin_Time"].tolist(),
#               data["4_kin_Energy"].tolist(), c='#A000FF', s=2)
# ax[0].scatter(data["6_kin_Time"].tolist(),
#               data["6_kin_Energy"].tolist(), c='#35FF19', s=2)
# ax[0].scatter(data["8_kin_Time"].tolist(),
#               data["8_kin_Energy"].tolist(), c='#220DFF', s=2)

# ax[1].scatter(data["2_pon_Time"].tolist(),
#               data["2_pon_Energy"].tolist(), c='#FFDD19', s=2)
# ax[1].scatter(data["4_pon_Time"].tolist(),
#               data["4_pon_Energy"].tolist(), c='#A000FF', s=2)
# ax[1].scatter(data["6_pon_Time"].tolist(),
#               data["6_pon_Energy"].tolist(), c='#35FF19', s=2)
# ax[1].scatter(data["8_pon_Time"].tolist(),
#               data["8_pon_Energy"].tolist(), c='#220DFF', s=2)

# ax[0].set_xlabel('Time (ps)')
# ax[1].set_xlabel('Time (ps)')
# ax[0].set_ylabel('Energy (eV)')
# ax[1].set_ylabel('Energy (eV)')
# ax[0].grid()
# ax[1].grid()
# ax[0].title.set_text('Kinetic')
# ax[1].title.set_text('Potential')
# plt.legend(['0.087 V/nm', '0.173 V/nm', '0.260 V/nm', '0.346 V/nm'], markerscale=5.)
# plt.savefig(directory+'/'+'dependence.png')
"""Вычисление спектров энергий"""
# energies_psd_kin = np.abs(sp.fftpack.fft(data["2_kin_Energy"].tolist()))
# energies_psd_pon = np.abs(sp.fftpack.fft(data["2_pon_Energy"].tolist()))
# n = len(energies_psd_kin)
# window = sp.signal.windows.gaussian(n, std=5000, sym=False)
# module_w_kin = energies_psd_kin * window
# module_w_pon = energies_psd_pon * window
# b, a = sp.signal.butter(2, 0.05)
# filtered_kin = sp.signal.filtfilt(b, a, module_w_kin)
# filtered_pon = sp.signal.filtfilt(b, a, module_w_pon)
# fftFreq = sp.fftpack.fftfreq(len(data["8_kin_Energy"].tolist()), 1/10**15)
# i = fftFreq > 0
# reverseCm = 1/((3*(10**10))/(fftFreq[i]))

# plt.plot(reverseCm, filtered_kin[i], c = 'darkblue')
# plt.plot(reverseCm, filtered_pon[i], c = 'red')
# plt.grid()
# plt.xlabel("Frequency (cm$^{-1}$)")
# plt.ylabel("Amplitude (a.u.)")
# plt.legend(["Kinetic", "Potential"])
# plt.title("0.087 V/nm")
# plt.xlim(13500, 16550)
# plt.savefig(directory+'/'+'dependence_087.png')
