import re
import Ui_design
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series
from scipy import fftpack
from PyQt5 import QtWidgets
import xlrd
import csv
from PyQt5.QtWidgets import QFileDialog
import subprocess
from matplotlib import pyplot as plt
'''
This programm create graph
Spectral density (Electric Field Amplitude).
Need to select a directoty with folders of Electric Field Amplitudes.
'''


#-----------------------------------------------------------------------------------------------------------------------
class MainApplication(QtWidgets.QMainWindow, Ui_design.Ui_PacketProcessor):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.directory = False
        self.allData = pd.DataFrame()
        self.changeDir_btn.clicked.connect(self.changeDir)
        self.process_btn.clicked.connect(self.goProcess)
        self.path_label.setText("Надо выбрать папку с папками")
#-----------------------------------------------------------------------------------------------------------------------
    def changeDir(self):
        self.path = QFileDialog.getExistingDirectory(self, 'Select a directory')
        os.chdir(self.path)
        self.path_label.setText(str(self.path))
#-----------------------------------------------------------------------------------------------------------------------
    def goProcess(self):
        self.folders = os.listdir(self.path)
        self.firstFolder = self.folders[0]
        self.tsDir = self.path + '/' + self.firstFolder
        #Take frequency for all data
        ts = pd.read_csv(self.tsDir + './spec.dat', delimiter=' ', index_col=None)
        ts.rename(columns = {'0.0' : 'Freq', '0.0.1' : 'Amplitude'}, inplace = True)
        #Write frequency to all Data
        self.allData['Frequency'] = ts['Freq']
        #Make array for plot
        frequencies = np.array(self.allData['Frequency'].to_list())
        with pd.ExcelWriter(str(self.path)+"/"+'result.xlsx') as writer:
           self.allData.to_excel(writer, sheet_name='first_exp', index=None, index_label=None)
        #Read all csv for amplitudes obtain
        for folder in self.folders:
            if os.path.isdir(self.path+'/'+folder):
                df = pd.read_csv(folder + './spec.dat', delimiter=' ', index_col=None)
                df.rename(columns = {'0.0' : 'Freq', '0.0.1' : 'Amplitude'}, inplace = True) 
                self.allData[os.path.basename(folder)] = df['Amplitude']
                amplitudes = np.array(df['Amplitude'].to_list())
                #Make pictures
                plt.gcf().clear()
                plt.plot(frequencies, amplitudes)
                plt.ylabel('Spectral density (a.u.)')
                plt.xlabel('Frequency ($cm^{-1}$)')
                plt.grid()
                plt.savefig(str(self.path)+'/'+str(folder)+'.png')
                #write amplitudes into the ./xlsx file
                with pd.ExcelWriter(str(self.path)+"/"+'result.xlsx') as writer:
                    self.allData.to_excel(writer, sheet_name='first_exp', index=None, index_label=None)
               

        
        data_xls = pd.read_excel(str(self.path)+"/"+'result.xlsx', 'first_exp', dtype=str, index_col=None)
        data_xls.to_csv(str(self.path)+"/" + 'result.csv', encoding='utf-8', index=False)
        
        self.df = pd.read_csv(str(self.path)+"/" + 'result.csv', sep=",", index_col=None, header=None)
        self.fieldAmp = self.df.iloc[0].to_list()
        self.fieldAmp.pop(0)
        self.fieldAmp = np.array(self.fieldAmp)/10

        self.amp = self.df.iloc[50001].to_list()
        self.amp.pop(0)
        self.amp = np.array(self.amp)
        
        print(self.fieldAmp)
        print(self.amp)
        self.graphicsView.setBackground('w')
        self.graphicsView.setLabel('bottom', 'Electric Field Amplitude', units='kcal/(mol*A*e)')
        self.graphicsView.setLabel('left', 'Spectral density', units="a.u.")
        self.graphicsView.plot(self.fieldAmp, self.amp, pen='r')

        
# ----------------------------------------------------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApplication()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()