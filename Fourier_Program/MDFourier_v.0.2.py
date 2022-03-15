import re
import pandas
import Fourier
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
class MainApplication(QtWidgets.QMainWindow, Fourier.Ui_MDFourier):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.directory = False
        self.data = None
        self.times = []
        self.energies = []
        self.uploadbtn.clicked.connect(self.upload)
        self.showBtn.clicked.connect(self.goProcess)
        self.saveBtn.clicked.connect(self.saveData)
        self.extBtn.clicked.connect(self.exitMode)
        self.opendatlogBtn.clicked.connect(self.openFiles)
        self.openCSVBtn.clicked.connect(self.fastCsv)
        self.openDCDBtn_2.clicked.connect(self.dcdproc)
        self.vmdBtn.clicked.connect(self.openvmd)
        #----------------------------------------------------------------------------------------------------------------------
    def openFiles(self):
        self.graphicsView_fourier.clear()
        self.graphicsView_energy.clear()
        self.times = []
        self.energies = []
        self.newTimes = []
        self.newEnergies = []
        self.frames = []
        self.dipMoment = []
        self.datafile = None
        self.logfile = None
        self.csvfile = None
        self.dcdfile = None
        self.atomLabel.setText("----------------------------")
        self.srLabel.setText("----------------------------")
        self.durLabel.setText("----------------------------")
        self.mlLabel.setText("----------------------------")
        self.tsLabel.setText("----------------------------")
        self.datLabel.setText("----------------------------")
        self.logLabel.setText("----------------------------")
        self.logLabel_2.setText("----------------------------")
        self.datLabel_2.setText("----------------------------")
        self.directory = QtWidgets.QFileDialog.getExistingDirectory(self)
        self.files = os.listdir(self.directory)
        for file in self.files:
            filename, file_extension = os.path.splitext(self.directory + "/" + file)
            if file_extension == ".dat":
                self.datafile =os.path.splitext(self.directory + "/" + file)[0]+".dat"
                if os.path.isfile(self.datafile):
                    if len(self.datafile) > 50:
                        self.datLabel.setText(str(os.path.basename(self.datafile)))
                    else:self.datLabel.setText(self.datafile)
                else:
                    self.datLabel.setText("File not found")
            if file_extension == ".log":
                self.logfile = os.path.splitext(self.directory + "/" + file)[0]+".log"
        if os.path.isfile(self.logfile):
            if len(self.datafile) > 50:
                self.logLabel.setText(str(os.path.basename(self.logfile)))
            else:
                self.logLabel.setText(self.logfile)
        else:
            self.logLabel.setText("File not found")
        os.chdir(self.directory)
        self.setWindowTitle("MDFourier in "+self.directory)
#-----------------------------------------------------------------------------------------------------------------------
    def fastCsv(self):
        self.graphicsView_fourier.clear()
        self.graphicsView_energy.clear()
        self.times = []
        self.energies = []
        self.newTimes = []
        self.newEnergies = []
        self.frames = []
        self.dipMoment = []
        self.datafile = None
        self.logfile = None
        self.csvfile = None
        self.dcdfile = None
        self.atomLabel.setText("----------------------------")
        self.srLabel.setText("----------------------------")
        self.durLabel.setText("----------------------------")
        self.mlLabel.setText("----------------------------")
        self.tsLabel.setText("----------------------------")
        self.datLabel.setText("----------------------------")
        self.logLabel.setText("----------------------------")
        self.logLabel_2.setText("----------------------------")
        self.datLabel_2.setText("----------------------------")
        self.csvfile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')[0]
        self.datLabel_2.setText(os.path.basename(self.csvfile))
#-----------------------------------------------------------------------------------------------------------------------
    def dcdproc(self):
        self.graphicsView_fourier.clear()
        self.graphicsView_energy.clear()
        self.times = []
        self.energies = []
        self.newTimes = []
        self.newEnergies = []
        self.frames = []
        self.dipMoment = []
        self.datafile = None
        self.logfile = None
        self.csvfile = None
        self.dcdfile = None
        self.atomLabel.setText("----------------------------")
        self.srLabel.setText("----------------------------")
        self.durLabel.setText("----------------------------")
        self.mlLabel.setText("----------------------------")
        self.tsLabel.setText("----------------------------")
        self.datLabel.setText("----------------------------")
        self.logLabel.setText("----------------------------")
        self.logLabel_2.setText("----------------------------")
        self.datLabel_2.setText("----------------------------")
        self.dcdfile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')[0]
        self.logLabel_2.setText(os.path.basename(self.dcdfile))
#-----------------------------------------------------------------------------------------------------------------------
    def upload(self):
        self.sampleRate = None
        self.graphicsView_energy.clear()
        self.times = []
        self.energies = []
        try:
            with open(self.logfile, 'r') as file:
                line = file.readlines()
                for i in line:
                    if re.findall(r'Info: \d+ ATOMS\n', i):
                        ions = re.search(r'\d+', i)
                        ions = int(ions.group(0))
                self.atomLabel.setText(str(ions))
        except TypeError:
            pass
        #---------------------------------------------------------------------------------------------------------------
        try:
            with open(self.logfile, 'r') as file:
                lastLine = file.readlines()[-5]
                duration = re.findall(r'\d+', lastLine)
                self.duration = int(duration[0])
                NS = round(float(self.duration/1000000), 3)
            self.durLabel.setText(str(NS) + " ns")
        except TypeError:
            pass
        try:
            with open(self.datafile, 'r') as f:
                my_lines = f.readlines()
            for i in my_lines:
                raw_time = re.findall(r'\d+ ', i)
                for a in raw_time:
                    self.times.append(int(a))
                raw_energy = re.findall(r'(?<=\s).*\d+.\d+', i)
                for b in raw_energy:
                    self.energies.append(float(b) / ions * 0.0434)
            self.times = np.array(self.times)*(10**(-15))
            self.mlLabel.setText(str(len(self.times)))
            self.cutTime = float(self.times[1]-self.times[0])
            self.sampleRate = round(float(1/self.cutTime))
            self.tsLabel.setText(str(self.cutTime)+" s")
            self.srLabel.setText(str(float(self.sampleRate)/(10**12))+" THz")
            self.graphicsView_energy.setBackground('w')
            self.graphicsView_energy.setLabel('bottom', 'time', units='s')
            self.graphicsView_energy.setLabel('left', 'Energy', units='eV')
            self.graphicsView_energy.plot(self.times, self.energies, pen='b', )
        except TypeError:
            pass
        # ---------------------------------------------------------------------------------------------------------------
        try:
            self.newTimes=[]
            self.newEnergies=[]
            self.atoms = float(self.atomNumValue.value())
            self.df = pd.read_csv(self.csvfile)
            self.newTimes = self.df['TS']
            self.newEnergies = self.df['ENERGY']
            self.newTimes = np.array(Series.tolist(self.newTimes))*(10**(-15))
            self.newEnergies = np.array(Series.tolist(self.newEnergies))
            try:
                self.newEnergies = self.newEnergies / self.atoms * 0.0434
            except RuntimeWarning:
                pass
            self.cutTime = float(self.newTimes[1] - self.newTimes[0])
            self.sampleRate = float(round(float(1 / float(self.cutTime))))
            self.tsLabel.setText(str(self.cutTime) + " s")
            self.srLabel.setText(str(float(self.sampleRate)/(10**12))+" THz")
            self.mlLabel.setText(str(len(self.newTimes)))
            self.graphicsView_energy.setBackground('w')
            self.graphicsView_energy.setLabel('bottom', 'time', units='s')
            self.graphicsView_energy.setLabel('left', 'Energy', units='eV')
            self.graphicsView_energy.plot(self.newTimes, self.newEnergies, pen='b')
            self.directory = os.path.dirname(self.csvfile)
        except ValueError:
            pass
#-----------------------------------------------------------------------------------------------------------------------
        try:
            out = []
            file = open(self.dcdfile, 'r+')
            for line in file.readlines():
                newLine = ' '.join(line.split()) + '\n'
                out.append(newLine)
            file.close()
            out[0] = 'frame dip_x dip_y dip_z abs\n'
            newFile = open('./dipoles.csv', 'w')
            for string in out:
                newFile.write(string)
            newFile.close()
            df = pd.read_csv('./dipoles.csv', sep=' ')
            self.frames = np.array(df['frame'].tolist())
            self.dipMoment = np.array(df['abs'].tolist())
            self.graphicsView_energy.setBackground('w')
            self.graphicsView_energy.setLabel('bottom', 'frame', units=None)
            self.graphicsView_energy.setLabel('left', 'Dipole Moment', units='rel. u.')
            self.graphicsView_energy.plot(self.frames, self.dipMoment, pen='b')
            self.sampleRate = round(1/(float(self.srNumValue.value())*(10**(-15))))
            self.srLabel.setText(str(float(self.sampleRate)/(10**12))+" THz")
            self.mlLabel.setText(str(len(self.frames)))
            self.tsLabel.setText(str(float(self.srNumValue.value())*(10**(-15)))+" s")
            self.directory = os.path.dirname(self.dcdfile)
        except TypeError:
            pass
# ----------------------------------------------------------------------------------------------------------------------
    def goProcess(self):
        self.graphicsView_fourier.clear()
        self.xSamp = []
        self.ySamp = []
        if self.logfile is not None:
            self.xSamp = np.array(self.times)
            self.ySamp = np.array(self.energies)
        elif self.csvfile is not None:
            self.xSamp = np.array(self.newTimes)
            self.ySamp = np.array(self.newEnergies)
        else:
            self.xSamp = np.array(self.frames)
            self.ySamp = np.array(self.dipMoment)

        if self.gaussBox.isChecked():
            self.window = np.hanning(int(round(len(self.xSamp))))
            self.y_res = self.ySamp * self.window
            energies_fft = sp.fftpack.fft(np.array(self.y_res))
            self.energies_psd = np.abs(energies_fft)
            self.fftFreq = sp.fftpack.fftfreq(len(energies_fft), 1 / float(self.sampleRate))
            self.i = self.fftFreq > 0
            self.reverseCm = 1/((3*(10**10))/(self.fftFreq[self.i]))
            self.graphicsView_fourier.setBackground('w')
            self.graphicsView_fourier.setLabel('bottom', 'k', units='cm^-1')
            self.graphicsView_fourier.setLabel('left', 'Amplitude', units=None)
            if self.naturalBox.isChecked():
                self.graphicsView_fourier.plot(self.reverseCm, self.energies_psd[self.i], pen='c')
            if self.logBox.isChecked():
                self.graphicsView_fourier.plot(self.reverseCm, np.log10(self.energies_psd[self.i]), pen='b')
            if self.tenLogsBox.isChecked():
                self.graphicsView_fourier.plot(self.reverseCm, 10 * np.log10(self.energies_psd[self.i]), pen='r')
        else:
            energies_fft = sp.fftpack.fft(np.array(self.ySamp))
            self.energies_psd = np.abs(energies_fft)
            self.fftFreq = sp.fftpack.fftfreq(len(energies_fft), 1 / float(self.sampleRate))
            self.i = self.fftFreq > 0
            self.graphicsView_fourier.setBackground('w')
            self.graphicsView_fourier.setLabel('bottom', 'k', units='cm^-1')
            self.graphicsView_fourier.setLabel('left', 'Amplitude', units=None)
            if self.naturalBox.isChecked():
                self.graphicsView_fourier.plot(self.fftFreq[self.i], self.energies_psd[self.i], pen='c')
            if self.logBox.isChecked():
                self.graphicsView_fourier.plot(self.fftFreq[self.i], np.log10(self.energies_psd[self.i]), pen='b')
            if self.tenLogsBox.isChecked():
                self.graphicsView_fourier.plot(self.fftFreq[self.i], 10 * np.log10(self.energies_psd[self.i]), pen='r')
# ----------------------------------------------------------------------------------------------------------------------
    def saveData(self):
        df = pandas.DataFrame()
        df['f (Hz)'] = (self.fftFreq[self.i])
        df['AmplitudePure'] = self.energies_psd[self.i]
        df['AmplitudeLog10'] = np.log10(self.energies_psd[self.i])
        df['Amplitude10Log10'] = 10 * np.log10(self.energies_psd[self.i])
        # te = pandas.DataFrame()
        # te['t (ns)'] = self.times
        # te['E (eV)'] = self.energ
        with pd.ExcelWriter(str(self.directory)+"/"+'result.xlsx') as writer:
            df.to_excel(writer, sheet_name='fft', index=None, index_label=None)
            # te.to_excel(writer, sheet_name='energy', index=None, index_label=None)
        self.graphicsView_fourier.clear()
        self.graphicsView_energy.clear()
        self.times = []
        self.energies = []
        self.newTimes = []
        self.newEnergies = []
        self.frames = []
        self.dipMoment = []
        self.atomLabel.setText("----------------------------")
        self.srLabel.setText("----------------------------")
        self.durLabel.setText("----------------------------")
        self.mlLabel.setText("----------------------------")
        self.tsLabel.setText("----------------------------")
        self.datafile = None
        self.logfile = None
        self.datLabel.setText("----------------------------")
        self.logLabel.setText("----------------------------")
        self.directory = None
# ----------------------------------------------------------------------------------------------------------------------
    def exitMode(self):
        exit()
# ----------------------------------------------------------------------------------------------------------------------
    def openvmd(self):
        subprocess.run(['vmd'])
# ----------------------------------------------------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApplication()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()