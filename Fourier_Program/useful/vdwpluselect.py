import re
import pandas
import desing2
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series
from scipy import fftpack
from PyQt5 import QtWidgets
import subprocess
from PyQt5.QtWidgets import QMessageBox
#-----------------------------------------------------------------------------------------------------------------------
class MainApplication(QtWidgets.QMainWindow, desing2.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.directory = False
        self.data = None
        self.workbtn.clicked.connect(self.work)


    def work(self):
        i = 1
        self.big_df = pd.DataFrame()
        self.path = "d://namd//variuosProteins//ff"
        self.dir = os.chdir(self.path)
        files = os.listdir()
        for file in files:
            filename, file_extension = os.path.splitext(str(dir) + "/" + file)
            if file_extension == ".csv":
                df = pd.read_csv(file)
                # big_df['TS']=df['TS']
                self.big_df['ELECT' + str(i)] = np.array(df['ELECT'].tolist()) / 4131 * 0.0434 * 1000
                i += 1
        # average
        # mean = big_df.mean(axis=1)
        # big_df['AVERAGE']=big_df.mean(axis=1)
        # Sum
        self.big_df['SUM'] = self.big_df.sum(axis=1)
        self.sumEnergies = np.array(self.big_df['SUM'].tolist())
        file = '4.csv'
        # get times
        self.tsdf = pd.read_csv(file)
        self.times = self.tsdf['TS']
        self.times = np.array(Series.tolist(self.times)) * (10 ** (-15))
        self.cutTime = float(self.times[1] - self.times[0])
        self.sampleRate = round((float(1 / self.cutTime)))

        self.energy1 = np.array(self.big_df['ELECT1'].tolist())
        self.window = np.hanning(int(round(len(self.times))))
        self.y_res1 = self.energy1 * self.window
        self.energies_fft1 = sp.fftpack.fft(np.array(self.y_res1))
        self.energies_psd1 = np.abs(self.energies_fft1)
        self.fftFreq = sp.fftpack.fftfreq(len(self.energies_fft1), 1 / float(self.sampleRate))
        self.i = self.fftFreq > 0

        self.energy2 = np.array(self.big_df['SUM'].tolist())
        self.window = np.hanning(int(round(len(self.times))))
        self.y_res2 = self.energy2 * self.window
        self.energies_fft2 = sp.fftpack.fft(np.array(self.y_res2))
        self.energies_psd2 = np.abs(self.energies_fft2)
        self.fftFreq = sp.fftpack.fftfreq(len(self.energies_fft2), 1 / float(self.sampleRate))
        self.i = self.fftFreq > 0

        self.graphicsView_1.setBackground('w')
        self.graphicsView_1.setLabel('bottom', 'time', units='s')
        self.graphicsView_1.setLabel('left', 'Energy', units='eV')
        self.graphicsView_1.plot(self.times, self.sumEnergies, pen='r')

        self.graphicsView_2.setBackground('w')
        self.graphicsView_2.setLabel('bottom', 'Frequency', units='Hz')
        self.graphicsView_2.setLabel('left', 'Amplitude', units=None)
        self.graphicsView_2.plot(self.fftFreq[self.i], self.energies_psd2[self.i], pen='r')



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApplication()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()