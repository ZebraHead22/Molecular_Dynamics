import re
import os
import ui
import sys
import numpy as np
import scipy as sp
from scipy import fftpack
from numpy.fft import fft, ifft
from PyQt5 import QtWidgets
from scipy.fft import rfft, rfftfreq

class MainApplication(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.plot_btn.clicked.connect(self.plot)
        self.coeff_dial.valueChanged.connect(self.your_update_function)
        self.your_update_function(self.coeff_dial.value())

    def your_update_function(self, value):
        self.time_wid.clear()
        self.spec_wid.clear()
        self.lcdNumber.display(value)

        self.c1 = self.c1_dsb.value()
        self.c2 = self.c2_dsb.value()
        self.freq = self.freq_dsb.value()*(10**12)
        self.freq2 = self.freq2_dsb.value()*(10**12)
        self.freq3_dsb = self.freq3_dsb_dsb.value()*(10**12)
        self.add_fun = self.coeff_dial.value()
    
        SR = 10**18        # No. of samples per second, 
        Ts = 1./SR          # Sampling interval 
        self.t = np.arange(0, 0.5*10**(-12), Ts)
        
        f = self.add_fun*np.sin(2*np.pi*self.freq3_dsb*self.t) #Функция для флуктуации
        signal = self.c1*np.sin(2*np.pi*self.freq*self.t+f)
        signal2 = self.c2*np.cos(2*np.pi*self.freq2*self.t+f)
        if self.c2 == 0:
            self.s = signal
        else:
            self.s = signal-signal2
       
        self.time_wid.setLabel('bottom', 'time', units='s')
        self.time_wid.setLabel('left', 'Amplitude (a.u.)', units=None)
        self.time_wid.plot(self.t, self.s, pen='r')

        self.X = fft(self.s)
        N = len(self.X)
        n = np.arange(N)
        T = N/SR
        self.freq = n/T 

        self.spec_wid.setLabel('bottom', 'k', units='Hz')
        self.spec_wid.setLabel('left', 'Amplitude', units='a.u.')
        self.spec_wid.setXRange(0, 10**15, padding=0)
        self.spec_wid.plot(self.freq, np.abs(self.X), pen='r')
    #-------------------------------------------------------------------------------
    def plot(self):
        self.time_wid.clear()
        self.spec_wid.clear()

        self.c1 = self.c1_dsb.value()
        self.c2 = self.c2_dsb.value()
        self.freq = self.freq_dsb.value()*(10**12)
        self.freq2 = self.freq2_dsb.value()*(10**12)
        self.freq3_dsb = self.freq3_dsb_dsb.value()*(10**12)
        self.add_fun = self.coeff_dial.value()/10
    
        SR = 10**18        # No. of samples per second, 
        Ts = 1./SR          # Sampling interval 
        self.t = np.arange(0, 10**(-12), Ts)
        
        f = self.add_fun*np.cos(2*np.pi*self.freq3_dsb*self.t) #Функция для флуктуации
        signal = self.c1*np.sin(2*np.pi*self.freq*self.t+f)
        signal2 = self.c2*np.cos(2*np.pi*self.freq2*self.t+f)
        if self.c2 == 0:
            self.s = signal
        else:
            self.s = signal*signal2
       
        self.time_wid.setLabel('bottom', 'time', units='s')
        self.time_wid.setLabel('left', 'Amplitude (a.u.)', units=None)
        self.time_wid.plot(self.t, self.s, pen='r')

        self.X = fft(self.s)
        N = len(self.X)
        n = np.arange(N)
        T = N/SR
        self.freq = n/T 

        self.spec_wid.setLabel('bottom', 'k', units='Hz')
        self.spec_wid.setLabel('left', 'Amplitude', units='a.u.')
        self.spec_wid.setXRange(0, 10**15, padding=0)
        self.spec_wid.plot(self.freq, np.abs(self.X), pen='r')

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApplication()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
