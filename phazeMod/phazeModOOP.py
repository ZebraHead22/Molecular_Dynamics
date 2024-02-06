import re
import os
import ui
import sys
import random
import numpy as np
import scipy as sp
from scipy import fftpack
from PyQt5 import QtWidgets
from numpy.fft import fft, ifft
from scipy.fft import rfft, rfftfreq

class MainApplication(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    "Equation parameters"
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        #Привязываем кнопку
        self.plot_btn.clicked.connect(self.plot)

    def plot(self): #Строит график
        #Определение параметров уравнения
        self.c1 = self.c1_dsb.value()
        self.c2 = self.c2_dsb.value()
        self.c3 = self.c3_dsb.value()
        self.sum = self.sum_dsb.value()

        self.freq = self.freq_dsb.value()*(10**12)
        self.freq2 = self.freq2_dsb.value()*(10**12)
        self.freq3 = self.freq3_dsb.value()*(10**12)
    
        SR = 10**18        # No. of samples per second, 
        Ts = 1./SR          # Sampling interval 
        self.t = np.arange(0, 0.5*10**(-12), Ts)
        
        o = self.c3*np.sin(2*np.pi*self.freq3*self.t) #Функция для флуктуации
        signal_main = self.c1*np.sin(2*np.pi*self.freq*self.t+o)
        signal_additional = self.c2*np.cos(2*np.pi*self.freq2*self.t+o)
        
        if self.sum == 1:
            if self.c2 == 0:
                self.s = signal_main
            else:
                self.s = signal_main - signal_additional
            print(self.s)
        
        elif self.sum >= 2:
            variables_names = {}
            for i in range(2, int(self.sum)+1):
                # dynamically create key
                key = 'member_' + str(i)
                # calculate value
                o = (random.random()*self.c3) * \
                    np.sin(2*np.pi*(random.random()*self.freq3)*self.t)
                value = ((random.random()*self.c1)*np.sin(2*np.pi*(random.random()*self.freq)*self.t+o)) - \
                    ((random.random()*self.c2)*np.cos(2*np.pi *
                    (random.random()*self.freq2)*self.t+o))
                variables_names[key] = value

            names_list = list(variables_names.values())
            summarize = 0
            for i in range(len(names_list)):
                summarize = summarize + names_list[i]
            additional_signals = np.array(summarize)
            self.s = (signal_main - signal_additional) + additional_signals

        else:
            raise TypeError ("Неверные данные")


        self.X = fft(self.s)
        N = len(self.X)
        n = np.arange(N)
        T = N/SR
        self.freq = n/T
        #Clear windows
        self.time_wid.clear()
        self.spec_wid.clear()
        #Plot ATR
        self.time_wid.setLabel('bottom', 'time', units='s')
        self.time_wid.setLabel('left', 'Amplitude', units='a.u.')
        self.time_wid.plot(self.t, self.s, pen='r')
        #Plot spectre
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
