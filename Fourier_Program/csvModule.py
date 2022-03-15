import re #import modules
import pandas
import Fourier
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series
from scipy import fftpack
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

def dolboeb(self):
    self.atoms = float(self.atoms.value())
    self.csvfile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')[0]
    self.termStatus.setText("WAIT")
    self.df = pd.read_csv(self.csvfile)
    self.newTimes = self.df['TS']
    self.newEnergies = self.df['ELECT']
    self.newTimes = np.array(Series.tolist(self.newTimes))
    self.newEnergies = np.array(Series.tolist(self.newEnergies))
    self.newTimes = self.newTimes / 1000000
    self.newEnergies = self.newEnergies / self.atoms * 0.0434
    cutTime = self.newTimes[1] - self.newTimes[0]
    self.graphicsView_energy.setBackground('w')
    self.graphicsView_energy.setLabel('bottom', 'time', units='ns')
    self.graphicsView_energy.setLabel('left', 'Energy', units='eV')
    self.graphicsView_energy.plot(self.newTimes, self.newEnergies, pen='b')

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    self.newy_gauss = np.array(
        gaussian(self.newTimes, float(self.newTimes[int(len(self.newTimes) / 2)]), 0.3))
    self.y_res = self.newEnergies * self.newy_gauss
    self.energies_fft = sp.fftpack.fft(np.array(self.y_res))
    self.energies_psd = np.abs(self.energies_fft)
    self.fftFreq = sp.fftpack.fftfreq(len(self.energies_fft), 1 / float(round(1 / cutTime * (10 ** 9))))
    self.i = self.fftFreq > 0
    self.fftFreqcm = 1 / ((2.9979 * (10 ** 10)) / self.fftFreq[self.i])
    self.graphicsView_fourier.setBackground('w')
    self.graphicsView_fourier.setLabel('bottom', 'Frequency (1/cm)', units=None)
    self.graphicsView_fourier.setLabel('left', 'Amplitude', units=None)
    self.graphicsView_fourier.plot(self.fftFreqcm, 10 * np.log10(self.energies_psd[self.i]), pen='r')